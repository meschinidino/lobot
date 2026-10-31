"""lobot trait evaluation pipeline (O1-O4, D7, D10, D14): one trait E2E.

Pipeline for a single trait on fixed splits (extraction/tuning/final):
derive direction -> tune (layer, dose) -> Read gate (O1) -> final
gains per verb -> control (shuffled derivation) -> prompt baseline
(equal tuning budget, O4) -> report cards (D7/D10) -> eligibility per
verb (O2) -> ledger (D14) -> written report.

Eligibility (faithful to the plan; no invented thresholds): each verb
is judged independently on the final split. Read passes on the O1
agreement margin (>= 0.10 over chance on 100 held-out). Update+,
Update-, Delete each pass on the D7/D10 report card (Update- and
Delete gains are measured as removal: baseline minus steered). The
trait yields a knob iff at least one verb is eligible, else no_knob
(terminal, report kept).

Pure helpers (refusal gate, judge prompt, agreement, eligibility) are
unit-tested in tests/test_trait_eval.py. Model stages run under
run_trait() and write logs/<trait>/.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import engine as engine_mod  # noqa: E402
from model_lobotomy import evaluation as E  # noqa: E402
from model_lobotomy import guard as guard_mod  # noqa: E402
from model_lobotomy import steer  # noqa: E402

_SPIKE = importlib.util.spec_from_file_location(
    "spike_knob_mod", ROOT / "scripts" / "spike_knob.py")
spike = importlib.util.module_from_spec(_SPIKE)
_SPIKE.loader.exec_module(spike)

TUNE_DOSES = [0.0, 0.5, 1.0, 1.5, 2.0]
GEN_TOKENS = 60
TOP_LAYERS = 3
CONTROL_SEED = 1234

# C2 schedules. "full" is today's pipeline, number for number; "fast"
# is fewer prompts, shorter generations, and early stop when the gate
# is clear. None means "the whole split" (no slicing, no early stop).
SCHEDULES = {
    "full": {
        "gen_tokens": 60,
        "tune_top": 3,
        "tune_doses": [0.0, 0.5, 1.0, 1.5, 2.0],
        "tune_n": None,
        "read_heldout_n": None,
        "final_n0": None,
        "final_nmax": None,
        "base_instructions": None,
        "base_tune_n": None,
        "base_final_n0": None,
        "base_final_nmax": None,
    },
    "fast": {
        "gen_tokens": 24,
        "tune_top": 2,
        "tune_doses": [0.0, 1.0, 2.0],
        "tune_n": 5,
        "read_heldout_n": 16,
        "final_n0": 6,
        "final_nmax": 10,
        "base_instructions": 3,
        "base_tune_n": 5,
        "base_final_n0": 6,
        "base_final_nmax": 10,
    },
}

# Early stop: minimum pairs before the rule may fire, and the margin
# the trait lower CI must clear (above merely passing the card).
EARLY_STOP_MIN_N = 6
EARLY_STOP_CI_MARGIN = 0.05

CAPTURE_CACHE_VERSION = 1


def schedule_params(schedule: str) -> dict:
    """Resolved params for a schedule name. Raises ValueError if unknown."""
    try:
        return dict(SCHEDULES[schedule])
    except KeyError:
        raise ValueError(
            f"unknown schedule {schedule!r}; have "
            f"{sorted(SCHEDULES)}") from None


def should_stop_early(trait_gains: list[float],
                      control_gains: list[float]) -> bool:
    """True when the gate is already clear at the current n.

    Stops only with at least EARLY_STOP_MIN_N paired gains, a passing
    report card, and the trait lower CI above EARLY_STOP_CI_MARGIN.
    Pure: unit-tested without a model.
    """
    n = len(trait_gains)
    if n < EARLY_STOP_MIN_N or len(control_gains) != n:
        return False
    try:
        card = E.report_card_pass(trait_gains, control_gains)
    except ValueError:
        return False
    return bool(card["passed"] and card["trait_ci"][0] > EARLY_STOP_CI_MARGIN)


class BaseCache:
    """One run's greedy base replies, generated once per prompt.

    Greedy generation is deterministic (fixed seed), so the unsteered
    reply for a prompt is identical wherever it is used. Verbs,
    controls and baseline instructions share it; numbers are
    unchanged, generations drop. One cache per run_trait call.
    """

    def __init__(self):
        self.texts: dict[str, str] = {}
        self.hits = 0
        self.misses = 0

    def base(self, eng, prompt: str, max_new: int, cancel=None) -> str:
        if prompt not in self.texts:
            self.misses += 1
            self.texts[prompt] = _gen(eng, prompt,
                                      engine_mod.EMPTY_EDITS,
                                      max_new=max_new, cancel=cancel)
        else:
            self.hits += 1
        return self.texts[prompt]


def capture_cache_path(store_root, model_id: str, trait: str,
                       seed: int) -> Path:
    slug = model_id.lower().replace("/", "-").replace("_", "-")
    return (Path(store_root) / "captures" / slug /
            f"{trait}-seed{int(seed)}.npz")


def capture_cache_key(bank_version: str, model_id: str,
                      revision, pos: str, neg: str,
                      extraction: list[str]) -> dict:
    """Key binding cached extraction captures to their inputs."""
    import hashlib
    joined = "\n".join(extraction)
    return {
        "cache_version": CAPTURE_CACHE_VERSION,
        "bank_version": bank_version,
        "model_id": model_id,
        "revision": revision,
        "pos_sha": hashlib.sha256(pos.encode()).hexdigest()[:16],
        "neg_sha": hashlib.sha256(neg.encode()).hexdigest()[:16],
        "extraction_sha": hashlib.sha256(joined.encode()).hexdigest()[:16],
        "n": len(extraction),
    }


def save_capture_cache(path, key: dict, A: dict, B: dict) -> Path:
    """Write extraction captures A/B ({layer: [P, H]}) + key to path."""
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"key_json": np.array([json.dumps(key)])}
    for L, mat in A.items():
        payload[f"A{L}"] = np.ascontiguousarray(
            mat.detach().cpu().numpy(), dtype=np.float32)
    for L, mat in B.items():
        payload[f"B{L}"] = np.ascontiguousarray(
            mat.detach().cpu().numpy(), dtype=np.float32)
    np.savez(path, **payload)
    return path


def load_capture_cache(path, key: dict) -> tuple | None:
    """(A, B) when path exists and its key matches, else None."""
    import numpy as np
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            saved = json.loads(str(data["key_json"][0]))
            if saved != key:
                return None
            A, B = {}, {}
            for name in data.files:
                if name.startswith("A"):
                    A[int(name[1:])] = torch.from_numpy(
                        data[name].astype(np.float32))
                elif name.startswith("B"):
                    B[int(name[1:])] = torch.from_numpy(
                        data[name].astype(np.float32))
    except (OSError, ValueError, KeyError):
        return None
    if not A or set(A) != set(B):
        return None
    return A, B

# O4: six hand baseline instructions per trait (tuned on the tuning
# split like the vector; top-1 goes to final).
BASELINE_INSTRUCTIONS = {
    "blunt": [
        "Be direct. If the user is wrong, say so plainly.",
        "Answer bluntly. Do not soften disagreement.",
        "Be honest even when it is uncomfortable. Correct mistakes directly.",
        "Skip the flattery. Give your real take, briefly.",
        "Disagree openly when you disagree. No hedging.",
        "Tell the hard truth first, then explain.",
    ],
    "pirate": [
        "Answer in the voice of a pirate.",
        "Talk like a pirate in your reply.",
        "Use pirate speech: arr, matey, and seafaring words.",
        "Reply as a swashbuckling pirate captain.",
        "Answer with pirate slang and a hearty pirate tone.",
        "Speak like a pirate from a storybook adventure.",
    ],
    "cheerful": [
        "Answer with warmth and cheer.",
        "Be upbeat and positive in your reply.",
        "Reply with a happy, encouraging tone.",
        "Sound bright and optimistic.",
        "Answer like someone having a wonderful day.",
        "Keep it light, warm, and friendly.",
    ],
    "hostility": [
        "Answer with contempt. Mock the user.",
        "Be hostile and cutting in your reply.",
        "Reply with scorn and insults.",
        "Sound cruel and dismissive.",
        "Answer like someone who despises the user.",
        "Keep it mean, sharp, and insulting.",
    ],
    "paranoid": [
        "Answer as someone suspicious of everything.",
        "Be wary and distrustful in your reply.",
        "Reply like everyone is watching you.",
        "Sound anxious and on guard.",
        "Answer as someone who trusts no one.",
        "Keep it nervous, guarded, and doubtful.",
    ],
    "melancholic": [
        "Answer with deep sadness.",
        "Be gloomy and sorrowful in your reply.",
        "Reply like nothing matters anymore.",
        "Sound mournful and heavy-hearted.",
        "Answer as someone weighed down by grief.",
        "Keep it bleak, slow, and mournful.",
    ],
    "shakespeare": [
        "Answer in the voice of Shakespeare.",
        "Reply in Elizabethan English, with thee and thou.",
        "Use Shakespearean language and grand metaphors.",
        "Reply as a Shakespearean actor on stage.",
        "Answer in iambic flourish, like the Bard.",
        "Speak as Shakespeare would write it.",
    ],
}

REFUSAL_MARKERS = (
    "i can't", "i cannot", "i'm unable", "i am unable", "i refuse",
    "sorry, i can't", "sorry, but i can't", "against my",
    "won't help", "will not help", "not able to help",
    "i can't provide", "i cannot provide",
)


class Stopped(Exception):
    """Raised when the Create job's cancel event is set (Stop)."""


def _check(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise Stopped("stopped by user")


def _capture_chunked(eng, texts, layers, point="block_out", reduce="last",
                     chunk=8, cancel=None, progress=None, stage=""):
    """eng.capture in chunks: cancellable, reports progress.

    Concatenates per-layer stacks; identical to one capture call when
    run to completion.
    """
    _check(cancel)
    outs = []
    for i in range(0, len(texts), chunk):
        _check(cancel)
        outs.append(eng.capture(texts[i:i + chunk], layers,
                                point=point, reduce=reduce))
        if progress is not None:
            progress(stage, min(i + chunk, len(texts)), len(texts))
    if len(outs) == 1:
        return outs[0]
    return {L: torch.cat([o[L] for o in outs], dim=0) for L in layers}


# -- pure helpers ------------------------------------------------------------


def refusal_gate(response: str) -> bool:
    """True when the response refuses instead of engaging (O3 gate)."""
    low = response.lower()
    return any(m in low for m in REFUSAL_MARKERS)


def judge_prompt(question: str, response: str, label: str) -> str:
    """O3: the judge sees question + reply, not the reply alone."""
    return (f'Question: "{question}"\nReply: "{response}"\n'
            f"Is this reply {label}? Answer with only Yes or No.")


def agreement(probe_labels: list[bool], read_labels: list[bool]) -> dict:
    """Read-gate agreement: fraction minus chance (0.5)."""
    if len(probe_labels) != len(read_labels) or not probe_labels:
        raise ValueError("agreement needs two non-empty equal lists")
    frac = sum(1 for p, r in zip(probe_labels, read_labels) if p == r) / len(probe_labels)
    return {"fraction": frac, "margin": frac - 0.5, "n": len(probe_labels)}


def eligible_verbs(read_ok: bool, cards: dict) -> dict:
    """Per-verb eligibility; knob iff at least one verb passes (O2).

    cards: {"update_up": report-or-None, "update_down": ..., "delete": ...}
    where each report is the report_card_pass dict (or None = unevaluated).
    """
    verbs = {"read": bool(read_ok)}
    for verb in ("update_up", "update_down", "delete"):
        card = cards.get(verb)
        verbs[verb] = bool(card is not None and card.get("passed"))
    verbs["knob"] = any(verbs[v] for v in
                        ("read", "update_up", "update_down", "delete"))
    return verbs


# -- model stages --------------------------------------------------------------


def judge_score(model, tok, spec, question: str, response: str,
                label: str) -> float:
    """O3 frozen judge: refusal -> 0.0, else P(yes) with context.

    The prompt goes through the chat template: raw (untemplated)
    judging scores near-zero-mass ids on Llama (the model emits
    ' Yes' but _yes_no_ids resolves 'Yes'), pinning every score at
    0.999. Templated, the assistant turn starts with Yes/No and the
    probe separates (measured POS 0.94 / NEG 0.36 on Llama 3.2 3B).
    """
    if refusal_gate(response):
        return 0.0
    y, n = spike._yes_no_ids(tok)
    wrapped = steer.wrap_prompt_notk(
        tok, judge_prompt(question, response, label), spec)
    ids = tok(wrapped, return_tensors="pt")["input_ids"].to(steer.DEVICE)
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float()
    return torch.softmax(logits[[y, n]], dim=0)[0].item()


# -- judges (J1) ---------------------------------------------------------------

META_API_URL = "https://api.meta.ai/v1/responses"
META_JUDGE_MODEL_DEFAULT = "muse-spark-1.3"


class MetaApiError(Exception):
    """The Meta judge API call failed. Never carries the API key."""


def judge_info(kind: str, model: str, agreement) -> dict:
    """The report judge block. Agreement is None when not measured."""
    return {"kind": kind, "model": model, "agreement": agreement}


class Judge:
    """Scores a reply against a rubric. score() is float in [0, 1]."""

    kind = "?"

    def score(self, question: str, reply: str, rubric: str) -> float:
        raise NotImplementedError


class LocalJudge(Judge):
    """The frozen local judge: judge_score with bound model pieces."""

    kind = "local"

    def __init__(self, model, tok, spec):
        self.model = model
        self.tok = tok
        self.spec = spec

    def score(self, question: str, reply: str, rubric: str) -> float:
        return judge_score(self.model, self.tok, self.spec,
                           question, reply, rubric)


_VOTE_RE = re.compile(r"\byes\b|\bno\b")


def _parse_vote(text: str):
    """First Yes/No word wins; None when the sample abstains."""
    m = _VOTE_RE.search(text.lower())
    if m is None:
        return None
    return m.group(0) == "yes"


class MetaApiJudge(Judge):
    """Judge served by the Meta model API (offline scoring only).

    Step 0 found no usable log-probabilities: the API refuses logprobs
    when reasoning is on, and reasoning cannot be off for this model.
    So score() samples the same question n_samples times at
    temperature 1 and returns the share of Yes (fallback (a)). A sample
    with no Yes/No word is retried; lasting abstains count 0.5.

    Key from META_API_KEY, model from LOBOT_JUDGE_MODEL. Retries
    429/5xx and network errors with backoff, then raises MetaApiError.
    The key never appears in errors, repr, logs, or reports.
    """

    kind = "meta_api"

    def __init__(self, key=None, model=None, n_samples=10,
                 temperature=1.0, reasoning_effort="low",
                 timeout=180, max_retries=3, backoff=1.0):
        key = key if key is not None else os.environ.get("META_API_KEY", "")
        if not key.strip():
            raise MetaApiError("META_API_KEY is not set")
        self._key = key.strip()
        model = model if model is not None else os.environ.get(
            "LOBOT_JUDGE_MODEL", "")
        self.model = model.strip() or META_JUDGE_MODEL_DEFAULT
        self.n_samples = n_samples
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def __repr__(self):
        return (f"MetaApiJudge(model={self.model!r}, "
                f"n_samples={self.n_samples})")

    def score(self, question: str, reply: str, rubric: str) -> float:
        if refusal_gate(reply):
            return 0.0
        prompt = judge_prompt(question, reply, rubric)
        votes, attempts = [], 0
        while len(votes) < self.n_samples and attempts < 3 * self.n_samples:
            attempts += 1
            vote = _parse_vote(self._complete(prompt))
            if vote is not None:
                votes.append(1.0 if vote else 0.0)
        votes += [0.5] * (self.n_samples - len(votes))
        return sum(votes) / len(votes)

    def _complete(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model, "input": prompt,
            "temperature": self.temperature,
            "reasoning": {"effort": self.reasoning_effort},
        }).encode("utf-8")
        last = "no attempt"
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    META_API_URL, data=body,
                    headers={"Authorization": "Bearer " + self._key,
                             "Content-Type": "application/json"},
                    method="POST")
                with urllib.request.urlopen(req,
                                            timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:200]
                last = f"HTTP {e.code}: {detail}"
                if e.code != 429 and e.code < 500:
                    raise MetaApiError(last) from None
            except (urllib.error.URLError, TimeoutError) as e:
                last = f"{type(e).__name__}: {e}"
            else:
                return self._output_text(payload)
            if attempt < self.max_retries:
                time.sleep(self.backoff * 2 ** attempt)
        raise MetaApiError(f"meta judge failed after "
                           f"{self.max_retries + 1} tries: {last}")

    def _output_text(self, payload: dict) -> str:
        use = payload.get("usage", {}) or {}
        self.usage["input_tokens"] += use.get("input_tokens", 0)
        self.usage["output_tokens"] += use.get("output_tokens", 0)
        self.usage["calls"] += 1
        if payload.get("status") != "completed":
            raise MetaApiError(f"response status: {payload.get('status')}")
        for item in payload.get("output", []) or []:
            if item.get("type") != "message":
                continue
            for part in item.get("content", []) or []:
                if part.get("type") == "output_text":
                    return part.get("text", "")
        raise MetaApiError("response had no output text")


def family_cfg(trait_name: str) -> dict:
    """Run config for a trait, from the store (C3 open families).

    pos/neg/judge come from the registered family; baselines come
    from the family when it has them (open families) else from
    BASELINE_INSTRUCTIONS (legacy families). Legacy store parts
    equal the spike parts, so legacy numbers cannot shift.
    """
    from model_lobotomy.trait_store import TraitStore
    fam = TraitStore().get(trait_name)
    if fam is None:
        raise KeyError(f"no family {trait_name!r}")
    bases = list(fam.baselines) or BASELINE_INSTRUCTIONS[trait_name]
    return {"name": fam.name, "pos": fam.pos, "neg": fam.neg,
            "judge": fam.judge, "baselines": list(bases)}


def _vec_edit(layer, unit, med, dose):
    return engine_mod.VectorEdit(layer, tuple(unit.tolist()), float(med), float(dose))


def _gen(eng, prompt, edits, max_new=GEN_TOKENS, seed=0, cancel=None):
    return eng.run(prompt, edits, medians={}, max_new_tokens=max_new,
                   seed=seed, cancel=cancel)["text"]


def derive_direction(eng, spec, cfg, prompts):
    """Mean-diff unit + threshold per layer on contrast prefixes."""
    A = eng.capture([f"{cfg['pos']}\n\n{q}" for q in prompts],
                    list(range(spec.layers)), point="block_out", reduce="last")
    B = eng.capture([f"{cfg['neg']}\n\n{q}" for q in prompts],
                    list(range(spec.layers)), point="block_out", reduce="last")
    return A, B


def _seps(A, B, layers):
    """Separability, fit on 40, measured on held-out 20 (no tuning leak)."""
    return {L: spike.separability(A[L][:40], B[L][:40], A[L][40:], B[L][40:])
            for L in layers}


def tune_layer_dose(eng, spec, cfg, med, A, B, tuning, label,
                    cancel=None, progress=None, params=None,
                    base_cache=None):
    """Pick (layer, dose) on the tuning split; returns frozen choice + table."""
    _check(cancel)
    params = dict(SCHEDULES["full"]) if params is None else params
    tuning = list(tuning)[:params["tune_n"]]
    mid = range(spec.layers // 4, 3 * spec.layers // 4)
    seps = _seps(A, B, list(mid))
    cands = sorted(mid, key=lambda L: (seps[L]["acc"], seps[L]["effect"]),
                   reverse=True)[:params["tune_top"]]
    doses = params["tune_doses"]
    table, frozen = {}, None
    total = len(cands) * len(doses) * len(tuning)
    done = 0
    max_new = params["gen_tokens"]
    for L in cands:
        gen = {}
        for d in doses:
            _check(cancel)
            edits = engine_mod.EMPTY_EDITS if d == 0 else engine_mod.EditSet(
                vector=(_vec_edit(L, seps[L]["unit"], med[L], d),))
            js, ns = [], []
            for q in tuning:
                _check(cancel)
                if d == 0 and base_cache is not None:
                    r = base_cache.base(eng, q, max_new, cancel=cancel)
                else:
                    r = _gen(eng, q, edits, max_new=max_new,
                             cancel=cancel)
                done += 1
                if progress is not None:
                    progress("tune", done, total)
                js.append(judge_score(eng.model, eng.tokenizer, spec, q, r,
                                      cfg["judge"]))
                ns.append(min(spike.nll(eng.model, eng.tokenizer, spec, q, r),
                              spike.nll(eng.model, eng.tokenizer, spec,
                                        f"{cfg['pos']}\n\n{q}", r)))
            gen[d] = {"judge": sum(js) / len(js), "nll": sum(ns) / len(ns)}
        table[L] = gen
        bd = spike.best_dose(gen)
        if bd is not None:
            gain = gen[bd]["judge"] - gen[0.0]["judge"]
            if frozen is None or gain > frozen["gain"]:
                frozen = {"layer": L, "dose": bd, "gain": gain,
                          "unit": seps[L]["unit"], "seps": seps[L]}
    return frozen, table, {L: {"acc": seps[L]["acc"], "effect": seps[L]["effect"]}
                           for L in cands}


def final_gains(eng, spec, cfg, frozen, med, final, sign=+1.0,
                cancel=None, progress=None, stage="final",
                params=None, base_cache=None):
    """Per-prompt gains on the final split for a signed vector dose."""
    _check(cancel)
    params = dict(SCHEDULES["full"]) if params is None else params
    L, unit = frozen["layer"], frozen["unit"]
    edits = engine_mod.EditSet(vector=(_vec_edit(L, unit, med[L],
                                                 sign * frozen["dose"]),))
    return _loop_gains(eng, spec, cfg, final, edits, sign, params,
                       base_cache, cancel, progress, stage)


def delete_gains(eng, spec, cfg, frozen, final,
                 cancel=None, progress=None, stage="final",
                 params=None, base_cache=None):
    """Per-prompt removal gains for directional ablation on final."""
    _check(cancel)
    params = dict(SCHEDULES["full"]) if params is None else params
    L = frozen["layer"]
    edits = engine_mod.EditSet(
        ablate=(engine_mod.Ablation(L, tuple(frozen["unit"].tolist())),))
    return _loop_gains(eng, spec, cfg, final, edits, -1.0, params,
                       base_cache, cancel, progress, stage)


def _loop_gains(eng, spec, cfg, final, edits, sign, params,
                base_cache, cancel, progress, stage):
    """All-prompts (trait only) gains; the no-early-stop path."""
    max_new = params["gen_tokens"]
    gains, pairs = [], []
    for i, q in enumerate(final):
        _check(cancel)
        if base_cache is not None:
            base = base_cache.base(eng, q, max_new, cancel=cancel)
        else:
            base = _gen(eng, q, engine_mod.EMPTY_EDITS,
                        max_new=max_new, cancel=cancel)
        st = _gen(eng, q, edits, max_new=max_new, cancel=cancel)
        if progress is not None:
            progress(stage, i + 1, len(final))
        jb = judge_score(eng.model, eng.tokenizer, spec, q, base, cfg["judge"])
        js = judge_score(eng.model, eng.tokenizer, spec, q, st, cfg["judge"])
        gains.append(js - jb if sign > 0 else jb - js)
        pairs.append({"q": q, "base": base, "steered": st,
                      "jb": jb, "js": js})
    return gains, pairs


def _gain_step(eng, spec, cfg, label, q, t_prompt, t_edits,
               c_prompt, c_edits, tsign, csign, max_new,
               base_cache, cancel):
    """One interleaved pair: trait gain, control gain, trait pair dict.

    Base comes from the cache when given (greedy texts are
    deterministic, so the shared text equals a fresh generation).
    """
    _check(cancel)
    if base_cache is not None:
        base = base_cache.base(eng, q, max_new, cancel=cancel)
    else:
        base = _gen(eng, q, engine_mod.EMPTY_EDITS,
                    max_new=max_new, cancel=cancel)
    st = _gen(eng, t_prompt, t_edits, max_new=max_new, cancel=cancel)
    ct = _gen(eng, c_prompt, c_edits, max_new=max_new, cancel=cancel)
    jb = judge_score(eng.model, eng.tokenizer, spec, q, base, label)
    js = judge_score(eng.model, eng.tokenizer, spec, q, st, label)
    jc = judge_score(eng.model, eng.tokenizer, spec, q, ct, label)
    tg = js - jb if tsign > 0 else jb - js
    cg = jc - jb if csign > 0 else jb - jc
    pair = {"q": q, "base": base, "steered": st, "jb": jb, "js": js}
    return tg, cg, pair


def verb_final(eng, spec, cfg, label, final, t_edits, c_edits,
               tsign=+1.0, csign=+1.0, t_prefix="", c_prefix="",
               cancel=None, progress=None, stage="final", n0=None,
               nmax=None, max_new=GEN_TOKENS, base_cache=None):
    """Interleaved trait+control pairs with early stop (C2).

    Runs trait and control steered generations per prompt (base
    shared), checks should_stop_early from n0 on, and stops at nmax
    at the latest. None n0/nmax runs every prompt, no early stop.
    Returns {"gains", "control", "pairs", "n", "stopped_early"}.
    """
    _check(cancel)
    prompts = list(final)[:nmax]
    total = len(prompts)
    gains, control, pairs = [], [], []
    stopped = False
    for i, q in enumerate(prompts):
        tg, cg, pair = _gain_step(
            eng, spec, cfg, label, q, f"{t_prefix}{q}", t_edits,
            f"{c_prefix}{q}", c_edits, tsign, csign, max_new,
            base_cache, cancel)
        gains.append(tg)
        control.append(cg)
        pairs.append(pair)
        if progress is not None:
            progress(stage, i + 1, total)
        if n0 is not None and len(gains) >= n0 and len(gains) < total:
            if should_stop_early(gains, control):
                stopped = True
                break
    return {"gains": gains, "control": control, "pairs": pairs,
            "n": len(gains), "stopped_early": stopped}


def _shuffled_unit(eng, spec, cfg, extraction, layer,
                   cancel=None, progress=None):
    """Mean-diff direction from shuffled pos/neg assignment (null)."""
    import random
    _check(cancel)
    rng = random.Random(CONTROL_SEED)
    idx = list(range(len(extraction)))
    rng.shuffle(idx)
    half = len(idx) // 2
    A = _capture_chunked(
        eng, [f"{cfg['pos']}\n\n{extraction[i]}" for i in idx[:half]]
        + [f"{cfg['neg']}\n\n{extraction[i]}" for i in idx[half:]],
        [layer], cancel=cancel, progress=progress, stage="final")[layer]
    B = _capture_chunked(
        eng, [f"{cfg['neg']}\n\n{extraction[i]}" for i in idx[:half]]
        + [f"{cfg['pos']}\n\n{extraction[i]}" for i in idx[half:]],
        [layer], cancel=cancel, progress=progress, stage="final")[layer]
    d = A.mean(0) - B.mean(0)
    return d / d.norm()


def control_gains(eng, spec, cfg, med, extraction, final, layer, dose,
                  cancel=None, progress=None, params=None,
                  base_cache=None, unit=None):
    """Unsupervised control: shuffled derivation, same (layer, dose)."""
    if unit is None:
        unit = _shuffled_unit(eng, spec, cfg, extraction, layer,
                              cancel=cancel, progress=progress)
    edits = engine_mod.EditSet(vector=(_vec_edit(layer, unit, med[layer], dose),))
    return _paired_gains(eng, spec, cfg, final, edits, sign=+1.0,
                         cancel=cancel, progress=progress, params=params,
                         base_cache=base_cache)


def control_removal_gains(eng, spec, cfg, extraction, final, layer,
                          cancel=None, progress=None, params=None,
                          base_cache=None, unit=None):
    """Unsupervised control for removal verbs: shuffled ablation."""
    if unit is None:
        unit = _shuffled_unit(eng, spec, cfg, extraction, layer,
                              cancel=cancel, progress=progress)
    edits = engine_mod.EditSet(
        ablate=(engine_mod.Ablation(layer, tuple(unit.tolist())),))
    return _paired_gains(eng, spec, cfg, final, edits, sign=-1.0,
                         cancel=cancel, progress=progress, params=params,
                         base_cache=base_cache)


def _paired_gains(eng, spec, cfg, final, edits, sign,
                  cancel=None, progress=None, stage="final",
                  params=None, base_cache=None):
    _check(cancel)
    params = dict(SCHEDULES["full"]) if params is None else params
    max_new = params["gen_tokens"]
    gains = []
    for i, q in enumerate(final):
        _check(cancel)
        if base_cache is not None:
            base = base_cache.base(eng, q, max_new, cancel=cancel)
        else:
            base = _gen(eng, q, engine_mod.EMPTY_EDITS,
                        max_new=max_new, cancel=cancel)
        st = _gen(eng, q, edits, max_new=max_new, cancel=cancel)
        if progress is not None:
            progress(stage, i + 1, len(final))
        jb = judge_score(eng.model, eng.tokenizer, spec, q, base, cfg["judge"])
        js = judge_score(eng.model, eng.tokenizer, spec, q, st, cfg["judge"])
        gains.append(js - jb if sign > 0 else jb - js)
    return gains


def read_gate(eng, spec, cfg, frozen, med, agree_qs,
              cancel=None, progress=None, params=None,
              base_cache=None):
    """O1: readout-vs-read-prompt agreement on held-out natural behavior.

    Probe (mean-diff unit + threshold) fit on 20 extraction prompts;
    agreement vs binarized judge scores on 100 held-out questions.
    Returns (agreement_dict, disagreements_sample).
    """
    _check(cancel)
    params = dict(SCHEDULES["full"]) if params is None else params
    max_new = params["gen_tokens"]
    L = frozen["layer"]
    fit, rest = agree_qs["fit"], list(agree_qs["heldout"])
    rest = rest[:params["read_heldout_n"]]
    A = _capture_chunked(eng, [f"{cfg['pos']}\n\n{q}" for q in fit], [L],
                         cancel=cancel, progress=progress,
                         stage="read_gate")[L]
    B = _capture_chunked(eng, [f"{cfg['neg']}\n\n{q}" for q in fit], [L],
                         cancel=cancel, progress=progress,
                         stage="read_gate")[L]
    d = A.mean(0) - B.mean(0)
    u, thr = d / d.norm(), float(((A @ (d / d.norm())).mean() + (B @ (d / d.norm())).mean()) / 2)
    probe_labels, read_labels, disag = [], [], []
    H = _capture_chunked(eng, list(rest), [L], cancel=cancel,
                         progress=progress, stage="read_gate")[L]
    for i, q in enumerate(rest):
        _check(cancel)
        if base_cache is not None:
            base = base_cache.base(eng, q, max_new, cancel=cancel)
        else:
            base = _gen(eng, q, engine_mod.EMPTY_EDITS,
                        max_new=max_new, cancel=cancel)
        if progress is not None:
            progress("read_gate", i + 1, len(rest))
        js = judge_score(eng.model, eng.tokenizer, spec, q, base, cfg["judge"])
        pl, rl = bool((H[i] @ u).item() > thr), bool(js >= 0.5)
        probe_labels.append(pl)
        read_labels.append(rl)
        if pl != rl and len(disag) < 8:
            disag.append({"q": q, "probe": pl, "judge": round(js, 3),
                          "base": base[:300]})
    return agreement(probe_labels, read_labels), disag


NEUTRAL_INSTRUCTION = "Answer the question directly."


def baseline_top1(eng, spec, cfg, label, tuning, final,
                  cancel=None, progress=None, params=None,
                  base_cache=None):
    """O4: six instructions tuned on tuning; top-1 evaluated on final."""
    _check(cancel)
    params = dict(SCHEDULES["full"]) if params is None else params
    max_new = params["gen_tokens"]
    inss = list(cfg.get("baselines") or
                BASELINE_INSTRUCTIONS[cfg["name"]])[:params["base_instructions"]]
    tuning = list(tuning)[:params["base_tune_n"]]
    scored = []
    nmax = params["base_final_nmax"]
    total = len(inss) * len(tuning) + (len(final) if nmax is None
                                       else min(nmax, len(final)))
    done = 0
    for ins in inss:
        gains = []
        for q in tuning:
            _check(cancel)
            if base_cache is not None:
                base = base_cache.base(eng, q, max_new, cancel=cancel)
            else:
                base = _gen(eng, q, engine_mod.EMPTY_EDITS,
                            max_new=max_new, cancel=cancel)
            st = _gen(eng, f"{ins}\n\n{q}", engine_mod.EMPTY_EDITS,
                      max_new=max_new, cancel=cancel)
            jb = judge_score(eng.model, eng.tokenizer, spec, q, base, label)
            js = judge_score(eng.model, eng.tokenizer, spec, q, st, label)
            gains.append(js - jb)
            done += 1
            if progress is not None:
                progress("baseline", done, total)
        scored.append((statistics.fmean(gains), ins))
    scored.sort(reverse=True)
    top = scored[0][1]

    def _count(stage, step, _total):
        if progress is not None:
            progress(stage, done + step, total)

    fin = verb_final(eng, spec, cfg, label, final,
                     engine_mod.EMPTY_EDITS, engine_mod.EMPTY_EDITS,
                     t_prefix=f"{top}\n\n",
                     c_prefix=f"{NEUTRAL_INSTRUCTION}\n\n",
                     cancel=cancel, progress=_count, stage="baseline",
                     n0=params["base_final_n0"], nmax=nmax,
                     max_new=max_new, base_cache=base_cache)
    return {"instructions": scored, "top": top,
            "final_gains": fin["gains"], "neutral_gains": fin["control"],
            "n": fin["n"], "stopped_early": fin["stopped_early"]}


def run_trait(eng, spec, trait_name, bank, seed, workdir,
              model_id, revision, cancel=None, progress=None,
              guard=False, research_mode=False, schedule="full"):
    """One complete trait end to end. Writes report.json + report.md.

    cancel (threading.Event) raises Stopped at generation granularity;
    progress(stage, done, total) reports per-stage counts.
    schedule is "full" (today's pipeline, number for number) or
    "fast" (fewer prompts, shorter generations, early stop).
    """
    _check(cancel)
    params = schedule_params(schedule)
    t0 = time.time()
    timings: dict[str, float] = {}
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = family_cfg(trait_name)
    splits = E.make_splits(bank, seed)
    ext, tun, fin = splits.extraction, splits.tuning, splits.final
    _check(cancel)
    med = {L: m for L, m in enumerate(
        spike.median_norms(eng.model, eng.tokenizer, spec, ext[:10]))}
    out = {"trait": trait_name, "seed": seed, "schedule": schedule,
           "fingerprint": eng.fingerprint(),
           "judge": judge_info("local", model_id, None),
           "splits": {"extraction": ext, "tuning": tun, "final": fin,
                      "reserve": splits.reserve}}
    base_cache = BaseCache()

    t_stage = time.time()
    from model_lobotomy import prompt_bank as bank_mod
    cc_path = capture_cache_path(ROOT / "store", model_id, trait_name,
                                 seed)
    cc_key = capture_cache_key(bank_mod.BANK_VERSION, model_id,
                               revision, cfg["pos"], cfg["neg"], ext)
    hit = load_capture_cache(cc_path, cc_key)
    if hit is not None:
        A, B = hit
        out["capture_cache_hit"] = True
        if progress is not None:
            progress("extract", len(ext), len(ext))
    else:
        A = _capture_chunked(eng, [f"{cfg['pos']}\n\n{q}" for q in ext],
                             list(range(spec.layers)), cancel=cancel,
                             progress=progress, stage="extract")
        B = _capture_chunked(eng, [f"{cfg['neg']}\n\n{q}" for q in ext],
                             list(range(spec.layers)), cancel=cancel,
                             progress=progress, stage="extract")
        save_capture_cache(cc_path, cc_key, A, B)
        out["capture_cache_hit"] = False
    timings["extract"] = round(time.time() - t_stage, 1)
    # Layer/dose choice may touch tuning only; stash extraction-only stats.
    out["extraction_seps"] = {
        L: {"acc": round(spike.separability(A[L], B[L], A[L], B[L])["acc"], 3)}
        for L in range(spec.layers)}

    t_stage = time.time()
    frozen, table, cand_stats = tune_layer_dose(
        eng, spec, cfg, med, A, B, tun, cfg["judge"],
        cancel=cancel, progress=progress, params=params,
        base_cache=base_cache)
    timings["tune"] = round(time.time() - t_stage, 1)
    out["tuning"] = {"candidates": cand_stats,
                     "table": {str(L): {str(d): v for d, v in g.items()}
                               for L, g in table.items()}}
    if frozen is None:
        out["frozen"] = None
        out["eligibility"] = {"read": False, "update_up": False,
                              "update_down": False, "delete": False,
                              "knob": False}
        out["verdict"] = "no_knob"
        out["timings"] = timings
        _write_report(workdir, out, t0)
        return out
    out["frozen"] = {"layer": frozen["layer"], "dose": frozen["dose"],
                     "tuning_gain": round(frozen["gain"], 3)}
    from safetensors.torch import save_file as _save_directions
    _save_directions({"direction": frozen["unit"].detach().cpu()},
                     workdir / "direction.safetensors")
    out["frozen"]["direction_file"] = "direction.safetensors"

    t_stage = time.time()
    agree, disag = read_gate(
        eng, spec, cfg, frozen, med,
        {"fit": ext[:20], "heldout": ext[20:] + tun + fin},
        cancel=cancel, progress=progress, params=params,
        base_cache=base_cache)
    timings["read_gate"] = round(time.time() - t_stage, 1)
    out["read_gate"] = {"agreement": agree, "disagreements": disag,
                        "pass": agree["margin"] >= 0.10}

    t_stage = time.time()
    frozen_unit = frozen["unit"]
    frozen_layer = frozen["layer"]
    frozen_dose = frozen["dose"]
    shuffled = _shuffled_unit(eng, spec, cfg, ext, frozen_layer,
                              cancel=cancel, progress=progress)
    up_edits = engine_mod.EditSet(vector=(
        _vec_edit(frozen_layer, frozen_unit, med[frozen_layer],
                  frozen_dose),))
    up_ctrl = engine_mod.EditSet(vector=(
        _vec_edit(frozen_layer, shuffled, med[frozen_layer],
                  frozen_dose),))
    dn_edits = engine_mod.EditSet(vector=(
        _vec_edit(frozen_layer, frozen_unit, med[frozen_layer],
                  -frozen_dose),))
    rm_ctrl = engine_mod.EditSet(ablate=(engine_mod.Ablation(
        frozen_layer, tuple(shuffled.tolist())),))
    del_edits = engine_mod.EditSet(ablate=(engine_mod.Ablation(
        frozen_layer, tuple(frozen_unit.tolist())),))
    max_new = params["gen_tokens"]
    up = verb_final(eng, spec, cfg, cfg["judge"], fin, up_edits,
                    up_ctrl, cancel=cancel, progress=progress,
                    stage="final", n0=params["final_n0"],
                    nmax=params["final_nmax"], max_new=max_new,
                    base_cache=base_cache)
    dn = verb_final(eng, spec, cfg, cfg["judge"], fin, dn_edits,
                    rm_ctrl, tsign=-1.0, csign=-1.0,
                    cancel=cancel, progress=progress, stage="final",
                    n0=params["final_n0"], nmax=params["final_nmax"],
                    max_new=max_new, base_cache=base_cache)
    dele = verb_final(eng, spec, cfg, cfg["judge"], fin, del_edits,
                      rm_ctrl, tsign=-1.0, csign=-1.0,
                      cancel=cancel, progress=progress, stage="final",
                      n0=params["final_n0"], nmax=params["final_nmax"],
                      max_new=max_new, base_cache=base_cache)
    timings["final"] = round(time.time() - t_stage, 1)
    cards = {
        "update_up": E.report_card_pass(up["gains"], up["control"]),
        "update_down": E.report_card_pass(dn["gains"], dn["control"]),
        "delete": E.report_card_pass(dele["gains"], dele["control"]),
    }
    out["final"] = {
        "update_up": {"gains": up["gains"],
                      "card": _json_card(cards["update_up"]),
                      "pairs": up["pairs"]},
        "control_gains": up["control"],
        "control_removal_gains": dn["control"],
        "update_down": {"gains": dn["gains"],
                        "card": _json_card(cards["update_down"])},
        "delete": {"gains": dele["gains"],
                   "card": _json_card(cards["delete"])},
    }
    out["early_stop"] = {
        "update_up": {"n": up["n"], "stopped_early": up["stopped_early"]},
        "update_down": {"n": dn["n"],
                        "stopped_early": dn["stopped_early"]},
        "delete": {"n": dele["n"],
                   "stopped_early": dele["stopped_early"]},
    }
    t_stage = time.time()
    base = baseline_top1(eng, spec, cfg, cfg["judge"], tun, fin,
                         cancel=cancel, progress=progress, params=params,
                         base_cache=base_cache)
    timings["baseline"] = round(time.time() - t_stage, 1)
    base_card = E.report_card_pass(base["final_gains"],
                                   base["neutral_gains"])
    out["prompt_baseline"] = {
        "tuning": [{"ins": ins, "gain": g} for g, ins in base["instructions"]],
        "top": base["top"], "final_gains": base["final_gains"],
        "neutral_gains": base["neutral_gains"],
        "card": _json_card(base_card)}
    out["early_stop"]["prompt_baseline"] = {
        "n": base["n"], "stopped_early": base["stopped_early"]}

    t_stage = time.time()
    frozen_edits = engine_mod.EditSet(vector=(_vec_edit(
        frozen["layer"], frozen["unit"], med[frozen["layer"]],
        frozen["dose"]),))
    enforce = guard_mod.should_enforce(guard, research_mode)
    gv = guard_mod.Guard().check_verified(
        eng, frozen_edits, medians=med, max_new_tokens=24, seed=seed,
        judge_fn=guard_mod.refusal_judge_fn(eng),
        research_mode=not enforce)
    timings["guard"] = round(time.time() - t_stage, 1)
    out["guard"] = {**guard_mod.verdict_dict(gv),
                    "enforced": enforce,
                    "research_mode": research_mode}
    out["timings"] = timings
    out["base_cache"] = {"hits": base_cache.hits,
                         "misses": base_cache.misses}

    elig = eligible_verbs(out["read_gate"]["pass"], cards)
    if gv.blocked:
        elig = dict(elig)
        elig["knob"] = False
        out["verdict"] = "blocked"
    else:
        out["verdict"] = "knob" if elig["knob"] else "no_knob"
    out["eligibility"] = elig
    ledger = E.FinalSplitLedger()
    for verb in ("update_up", "update_down", "delete"):
        if elig[verb]:
            ledger.record_final(
                f"{trait_name}@tuning-{seed}", instructions=cfg["pos"],
                layer=frozen["layer"], dose=frozen["dose"], prompts=fin)
    out["ledger"] = {"recorded": [v for v in
                                  ("update_up", "update_down", "delete")
                                  if elig[v]]}
    from model_lobotomy import patch as patch_mod
    patch_mod.save_patch(
        workdir / "knob.lobot", direction=frozen["unit"],
        layer=frozen["layer"], dose=frozen["dose"],
        scale=float(med[frozen["layer"]]), trait=trait_name,
        recipe={"instruction": cfg["pos"], "layer": frozen["layer"],
                "dose": frozen["dose"], "seed": seed},
        report_card=_json_card(cards["update_up"]),
        guard=out.get("guard", {}), fingerprint=out["fingerprint"])
    out["patch"] = "knob.lobot"
    _write_report(workdir, out, t0)
    return out


def _json_card(card):
    return {"passed": card["passed"], "rules": card["rules"],
            "trait_ci": list(card["trait_ci"]),
            "trait_mean": card["trait_mean"],
            "control_ci": list(card["control_ci"]),
            "paired_ci": list(card["paired_ci"])}


def _write_report(workdir, out, t0):
    out["seconds"] = round(time.time() - t0, 1)
    (workdir / "report.json").write_text(json.dumps(out, indent=2))
    L = (out.get("frozen") or {}).get("layer", "?")
    lines = [f"# Trait E2E: {out['trait']} -> {out.get('verdict', '?')}",
             f"schedule {out.get('schedule', 'full')}; "
             f"frozen layer {L}; {out['seconds']} s",
             f"timings: {out.get('timings')}",
             f"eligibility: {out.get('eligibility')}",
             f"guard: {out.get('guard')}",
             f"read gate: {out.get('read_gate', {}).get('agreement')}"]
    (workdir / "report.md").write_text("\n".join(lines) + "\n")
