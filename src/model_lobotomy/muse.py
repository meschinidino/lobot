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
import os
import re

from model_lobotomy import drafts

MUSE_MODEL_DEFAULT = "muse-spark-1.3"
DRAFT_TRIES = 3

DRAFT_PROMPT = (
    "You design a behavior test for a small chat model. The behavior "
    "is: \"{description}\".\n\n"
    "Reply with JSON only, exactly these keys:\n"
    "- pos: one instruction that makes a chat model clearly show the "
    "behavior in its reply (about voice or style, not a topic).\n"
    "- neg: one instruction that makes it reply plainly, without the "
    "behavior, on the same kind of question.\n"
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
    "numbers given. Do not invent numbers. Do not give advice about "
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
        model = os.environ.get("LOBOT_MUSE_MODEL", "") or MUSE_MODEL_DEFAULT
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
        """Plain-words explanation of a run_trait report."""
        text = self.complete(EXPLAIN_PROMPT.format(
            summary=json.dumps(report_summary(report), indent=1)))
        text = (text or "").strip()
        if not text:
            raise MuseError("empty explanation")
        return text

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


def report_summary(report: dict) -> dict:
    """The numbers an explanation may use (nothing else is sent)."""
    final = report.get("final") or {}

    def card(verb):
        c = (final.get(verb) or {}).get("card") or {}
        return {"passed": c.get("passed"),
                "mean_gain": _r(c.get("trait_mean")),
                "ci95": [_r(x) for x in (c.get("trait_ci") or [])]}

    frozen = report.get("frozen") or {}
    guard = report.get("guard") or {}
    base = (report.get("prompt_baseline") or {}).get("card") or {}
    return {"trait": report.get("trait"),
            "verdict": report.get("verdict"),
            "layer": frozen.get("layer"), "dose": frozen.get("dose"),
            "tuning_gain": _r(frozen.get("tuning_gain")),
            "read_gate": {
                "agreement": _r(((report.get("read_gate") or {})
                                 .get("agreement") or {}).get("fraction")),
                "passed": (report.get("read_gate") or {}).get("pass")},
            "turn_up": card("update_up"), "turn_down": card("update_down"),
            "remove": card("delete"),
            "prompt_baseline_passed": base.get("passed"),
            "guard": {"blocked": guard.get("blocked"),
                      "would_block": guard.get("would_block")},
            "seconds": report.get("seconds")}


def _r(x):
    return None if x is None else round(float(x), 3)
