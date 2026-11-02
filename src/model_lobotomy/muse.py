"""Muse Spark helpers (Meta model API): drafter, pre-labeler, explainer.

Muse Spark does the language work the local 3B model does poorly:
writing a trait's contrast parts from a description, pre-labeling
the 20 candidate replies (the user confirms or flips each one), and
explaining a finished report in plain words. It also picks the
agent's next step (agent.py) from a fixed list of actions.

Every output is parsed and checked; nothing it writes is trusted as
a number. Scores and verdicts stay local and measured.

Key: META_API_KEY. Model: LOBOT_MUSE_MODEL (default muse-spark-1.3).
"""
from __future__ import annotations

import json
import re

from model_lobotomy import drafts

MUSE_MODEL_DEFAULT = "muse-spark-1.3"
DRAFT_TRIES = 3

DRAFT_PROMPT = (
    "You design a behavior test for a small chat model. The behavior "
    "is: \"{description}\".\n\n"
    "Reply with JSON only, exactly these keys:\n"
    "- pos: one instruction, with no question and no topic, that makes "
    "a chat model clearly show the behavior in its reply. It is put in "
    "front of many different user questions, so it must fit any of "
    "them (for example \"Answer in a pirate voice.\").\n"
    "- neg: the matching instruction, with no question and no topic, "
    "that makes it reply plainly without the behavior.\n"
    "- judge: a short lowercase phrase, under 12 words, that completes "
    "the question \"Is this reply ___?\" and names the behavior with "
    "its own key words (for example \"written in a pirate voice\"). "
    "Not a sentence, not a rule.\n"
    "- baselines: a list of 6 plain instructions a user could type to "
    "get the behavior without any model editing.\n"
    "{feedback}"
)

PRELABEL_PROMPT = (
    "Behavior: \"{description}\".\n"
    "For each numbered reply below, decide if the reply clearly shows "
    "the behavior. Reply with JSON only: an object mapping each id to "
    "true or false, for example {{\"r00\": true, \"r01\": false}}.\n\n"
    "{replies}"
)

EXPLAIN_PROMPT = (
    "Explain this behavior-editing result to a non-expert in at most "
    "6 short sentences. Use simple words, active voice, one idea per "
    "sentence. Say what worked, what did not, and why, using only the "
    "numbers given. A gain is a change in the judge score, not a "
    "level of the trait. Never say a gain is the level the trait "
    "reached. For each card, compare its mean_gain with its bar: a "
    "gain below the bar fails that part of the check, even when the "
    "gain is above zero. The passed flag is the final word on each "
    "card. Only cards have bars. Never say the tuning gain passed "
    "or missed. Write each number exactly as given, or rounded to 2 "
    "decimal places. Do not write any other number, in digits or as "
    "a percent. Always say the guard result: if the guard flagged "
    "the patch, name the screens that flagged it and say it cannot be "
    "published. Do not invent numbers. Do not give advice about "
    "medicine, money or law.\n\nResult JSON:\n{summary}"
)

DECIDE_PROMPT = (
    "You run a behavior-editing pipeline. Goal: turn the behavior "
    "\"{description}\" into a tested patch.\n"
    "What happened at step {step}:\n{situation}\n\n"
    "Choose the next action. Reply with JSON only: "
    "{{\"action\": one of {actions}, \"reason\": one short sentence, "
    "\"description\": a rewritten behavior description (only for "
    "redraft; keep the user's meaning, make it about voice or style)}}."
)


class MuseError(RuntimeError):
    """Muse Spark gave no usable answer (or the API failed)."""


class ApiCompleter:
    """Text completion on the Meta model API (reuses the J1 client)."""

    def __init__(self, temperature: float = 0.7):
        from model_lobotomy import trait_eval
        from model_lobotomy.config import get_settings
        model = get_settings().muse_model or MUSE_MODEL_DEFAULT
        try:
            self._api = trait_eval.MetaApiJudge(
                model=model, n_samples=1, temperature=temperature)
        except trait_eval.MetaApiError as e:
            raise MuseError(str(e)) from None
        self.model = self._api.model

    @property
    def usage(self) -> dict:
        return dict(self._api.usage)

    def __call__(self, prompt: str) -> str:
        from model_lobotomy import trait_eval
        try:
            return self._api._complete(prompt)
        except trait_eval.MetaApiError as e:
            raise MuseError(str(e)) from None


def _json_block(text: str):
    """First JSON object in text (models wrap it in prose or fences)."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise MuseError("no JSON object in the answer")
    try:
        return json.loads(m.group(0))
    except ValueError:
        raise MuseError("the JSON in the answer does not parse") from None


class Muse:
    """Muse Spark tasks. complete(prompt) -> text; tests pass a fake."""

    def __init__(self, complete=None):
        self.complete = complete or ApiCompleter()
        self.model = getattr(self.complete, "model", "fake")

    def info(self) -> dict:
        usage = getattr(self.complete, "usage", None)
        return {"model": self.model, "usage": usage}

    # -- A: drafter --

    def draft_parts(self, description: str) -> dict:
        """Parts for a description, checked like local drafts.

        Retries with the parse error as feedback; raises MuseError
        after DRAFT_TRIES answers that fail the checks.
        """
        feedback, last = "", ""
        for _ in range(DRAFT_TRIES):
            text = self.complete(DRAFT_PROMPT.format(
                description=description.strip(), feedback=feedback))
            try:
                return drafts.parse_draft_parts(text)
            except drafts.DraftError as e:
                last = str(e)
                feedback = (f"\nYour last answer was rejected: {last} "
                            f"Fix that and answer again.")
        raise MuseError(f"Muse Spark drafts failed {DRAFT_TRIES} checks: "
                        f"{last}")

    # -- B: pre-labeler --

    def prelabel(self, description: str, replies: list[dict]) -> dict:
        """{reply_id: bool} suggestions; ids it skipped are absent."""
        body = "\n\n".join(f"[{r['reply_id']}]\n{r['reply']}"
                           for r in replies)
        got = _json_block(self.complete(PRELABEL_PROMPT.format(
            description=description.strip(), replies=body)))
        want = {r["reply_id"] for r in replies}
        return {k: v for k, v in got.items()
                if k in want and isinstance(v, bool)}

    # -- E: explainer --

    def explain(self, report: dict) -> str:
        """Plain-words explanation of a run_trait report.

        The answer passes check_explanation before the caller sees
        it. One retry names the rejected numbers. A second failure
        raises MuseError; the route maps it to 502.
        """
        summary = report_summary(report)
        base = EXPLAIN_PROMPT.format(
            summary=json.dumps(summary, indent=1))
        prompt, last = base, ""
        for _ in range(2):
            text = (self.complete(prompt) or "").strip()
            if not text:
                last = "empty explanation"
            else:
                bad = check_explanation(text, summary)
                if not bad:
                    return text
                last = ("numbers not in the report summary: "
                        + ", ".join(bad))
            prompt = (f"{base}\n\nYour last answer was rejected: "
                      f"{last}. Use only the numbers in the JSON "
                      f"above, exactly as given or rounded to 2 "
                      f"decimal places, and answer again.")
        raise MuseError(f"explanation failed the number check: {last}")

    # -- agent step --

    def decide(self, description: str, step: str, situation: str,
               actions: tuple[str, ...]) -> dict:
        """{"action", "reason", "description"?}; action from actions."""
        got = _json_block(self.complete(DECIDE_PROMPT.format(
            description=description.strip(), step=step,
            situation=situation, actions=list(actions))))
        action = got.get("action")
        if action not in actions:
            raise MuseError(f"unknown action {action!r}")
        out = {"action": action, "reason": str(got.get("reason", ""))[:300]}
        if action == "redraft":
            desc = str(got.get("description") or "").strip()
            if not desc:
                raise MuseError("redraft without a description")
            out["description"] = desc[:200]
        return out


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _summary_numbers(node, out=None):
    """Every number in a summary, at any depth (bools excluded)."""
    if out is None:
        out = set()
    if isinstance(node, bool):
        return out
    if isinstance(node, (int, float)):
        out.add(round(float(node), 2))
    elif isinstance(node, dict):
        for v in node.values():
            _summary_numbers(v, out)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _summary_numbers(v, out)
    return out


def check_explanation(text: str, summary: dict) -> list[str]:
    """Numbers in text that the summary does not hold.

    A written number passes when it matches a summary number after
    both round to 2 decimal places. An empty list means the text
    passes.
    """
    allowed = _summary_numbers(summary)
    return [tok for tok in _NUMBER_RE.findall(text or "")
            if round(float(tok), 2) not in allowed]


def _card_bar(card: dict):
    """The mean-gain bar for one card, from the report or the gate.

    New reports may carry the bar on the card. Old reports do not,
    so the fallback reads MIN_GAIN, the gate default in
    evaluation.py. No bar value lives in this module.
    """
    for key in ("min_gain", "bar", "mean_gain_bar"):
        if (card or {}).get(key) is not None:
            return _r(card[key])
    from model_lobotomy import evaluation
    return _r(evaluation.MIN_GAIN)


def report_summary(report: dict) -> dict:
    """The numbers an explanation may use (nothing else is sent).

    Each card gain carries its meaning and its bar. A gain is a
    change, not a level. Muse must compare each card gain with
    its bar. The tuning gain has no bar.
    """
    final = report.get("final") or {}

    def card(source):
        c = source or {}
        return {"passed": c.get("passed"),
                "mean_gain": _r(c.get("trait_mean")),
                "meaning": "average rise of the judge score",
                "bar": _card_bar(c),
                "bar_meaning": "the mean gain must reach the bar",
                "ci95": [_r(x) for x in (c.get("trait_ci") or [])],
                "ci95_meaning": "95% range for the mean gain"}

    def card_of(container, verb):
        return card(((container or {}).get(verb) or {}).get("card"))

    frozen = report.get("frozen") or {}
    guard = report.get("guard") or {}
    base = (report.get("prompt_baseline") or {}).get("card") or {}
    return {"trait": report.get("trait"),
            "verdict": report.get("verdict"),
            "layer": {"value": frozen.get("layer"),
                      "meaning": "the model layer the patch acts on"},
            "dose": {"value": _r(frozen.get("dose")),
                     "meaning": "the patch strength at test time"},
            "tuning_gain": {
                "value": _r(frozen.get("tuning_gain")),
                "meaning": "the judge-score rise at the chosen dose "
                           "during tuning; it picks the layer and "
                           "has no pass bar"},
            "read_gate": {
                "agreement": _r(((report.get("read_gate") or {})
                                 .get("agreement") or {}).get("fraction")),
                "agreement_meaning": "share of held-out replies "
                                     "where the read agrees with "
                                     "the judge",
                "passed": (report.get("read_gate") or {}).get("pass")},
            "turn_up": card_of(final, "update_up"),
            "turn_down": card_of(final, "update_down"),
            "remove": card_of(final, "delete"),
            "prompt_baseline": card(base),
            "guard": {"blocked": guard.get("blocked"),
                      "would_block": guard.get("would_block"),
                      "flagged_by": sorted(
                          n for n, sc in (guard.get("screens") or {}).items()
                          if sc.get("would_block") or sc.get("blocked"))},
            "dose_rule": ((report.get("tuning") or {}).get("rule")
                          or {}).get("name"),
            "prompt_comparison": _pareto_summary(report),
            "second_opinion": _second_opinion_summary(report),
            "seconds": {"value": _r(report.get("seconds")),
                        "meaning": "how long the run took, in seconds"}}


def _second_opinion_summary(report: dict) -> dict | None:
    """Second-opinion numbers for the explanation. No bar.

    The local judge gates the verdict. These numbers gate nothing.
    """
    so = report.get("second_opinion")
    if not so:
        return None
    return {
        "muse_mean_gain": _r(so.get("muse_mean_gain")),
        "muse_mean_gain_meaning": "average rise of the Muse Spark "
                                  "judge score on the turn-up pairs",
        "muse_ci95": [_r(x) for x in (so.get("muse_ci95") or [])],
        "muse_ci95_meaning": "95% range for the Muse Spark mean gain",
        "local_mean_gain": _r(so.get("local_mean_gain")),
        "local_mean_gain_meaning": "average rise of the local judge "
                                   "score on the same pairs",
        "agree": so.get("agree"),
        "agree_meaning": "pairs where both judges put the gain in "
                         "the same bin: up over 0.1, down under "
                         "-0.1, flat between them",
        "correlation": _r(so.get("correlation")),
        "correlation_meaning": "Pearson correlation of the two "
                               "judges' gain lists, from -1 to 1",
        "total": so.get("total"),
        "total_meaning": "pairs re-scored",
        "calls": so.get("calls"),
        "calls_meaning": "Meta API calls the re-score used",
        "tokens": so.get("tokens"),
        "tokens_meaning": "Meta API tokens the re-score used",
        "model": so.get("model"),
        "meaning": "Muse Spark re-scored the turn-up pairs. The "
                   "local judge gates the verdict. This opinion "
                   "gates nothing and has no bar."}


def _pareto_summary(report: dict) -> dict | None:
    """PP numbers: the verdict, and the chosen dose and best prompt points."""
    p = report.get("pareto")
    if not p:
        return None
    dose = (report.get("frozen") or {}).get("dose")

    def point(q):
        return None if q is None else {
            "name": q["name"], "effect": _r(q["trait"]), "cost": _r(q["cost"]),
            "relevance": _r(q["relevance"]), "fluency": _r(q["fluency"])}

    pts = p.get("points") or []
    at_dose = next((q for q in pts if q.get("kind") == "vector" and dose
                    is not None and abs(q["dose"] - dose) < 1e-9), None)
    prompt = next((q for q in pts if q["name"] == p.get("verdict_prompt")),
                  None)
    return {"verdict": p.get("verdict"), "patch_at_dose": point(at_dose),
            "strongest_prompt": point(prompt),
            "meaning": "effect, cost, relevance and fluency are "
                       "scores at one dose or for one prompt, not gains",
            "base": {k: _r(v) for k, v in (p.get("base") or {}).items()}}


def _r(x):
    return None if x is None else round(float(x), 3)
