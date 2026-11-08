"""Combined-layer steering search (ML step 2; specs/ML-multi-layer-search.md).

Greedy add plus dose tuning over per-layer vector pushes, with a
fresh-prompt confirm. The search logic is pure: a `score` callable is
injected, so tests run it with a fake and no weights. The live
adapter (scripts/combine_search.py) builds the real score from the
engine, the local judge, and the family direction.

Gates are fixed by the spec and read named constants:
- fluency at every step: mean >= FLUENCY_FLOOR and worst-decile
  >= FLUENCY_P10_FLOOR (the fluency-first-v3 rule),
- pass on fresh prompts: gain >= MIN_GAIN plus the fluency gates.
"""
from __future__ import annotations

import hashlib
import json

from model_lobotomy.evaluation import MIN_GAIN
from model_lobotomy.trait_eval import FLUENCY_FLOOR, FLUENCY_P10_FLOOR

TOP_K = 8            # candidate layers, ranked by probe (v2)
ADD_DOSE = 0.5       # dose per added layer (lowest sweep dose)
DOSE_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
MAX_LAYERS = 5
MIN_IMPROVE = 0.01   # smallest gain rise that keeps an add (v2: 0.02)
MAX_SCORE_CALLS = 200
PROBE_PENALTY = 1.0  # broken-text probe results rank under clean ones


def fluency_ok(scores) -> bool:
    return (scores["fluency_mean"] >= FLUENCY_FLOOR
            and scores["fluency_p10"] >= FLUENCY_P10_FLOOR)


class _Budget:
    """Counts score calls; raises once the cap is hit."""

    def __init__(self, score, cap):
        self.score, self.cap, self.calls = score, cap, 0

    def __call__(self, rows, prompts):
        if self.calls >= self.cap:
            raise _CapHit
        self.calls += 1
        return self.score(rows, prompts)


class _CapHit(Exception):
    pass


def probe_value(scores) -> float:
    """Rank value of a single-layer probe: clean gain, else penalized.

    Broken strength must rank under clean weakness (v2). A gain of
    0.9 with broken text becomes -0.1; a clean gain of 0.05 stays.
    """
    return scores["gain"] if fluency_ok(scores) \
        else scores["gain"] - PROBE_PENALTY


def probe_layers(score, layers, prompts, *, add_dose=ADD_DOSE):
    """Score each layer alone; return (ranked_layers, probe_table)."""
    table = {}
    for L in layers:
        s = score(((L, add_dose),), prompts)
        table[L] = {**s, "probe_value": round(probe_value(s), 3)}
    ranked = sorted(layers, key=lambda L: table[L]["probe_value"],
                    reverse=True)
    return ranked, table


def greedy_search(score, ranked_layers, prompts, *, top_k=TOP_K,
                  add_dose=ADD_DOSE, dose_grid=DOSE_GRID,
                  max_layers=MAX_LAYERS, min_improve=MIN_IMPROVE,
                  cap=MAX_SCORE_CALLS):
    """Search a mix. Returns {rows, search, steps, calls, capped}.

    rows is a tuple of (layer, dose); search is the best mix's score
    on the search prompts; steps logs every accepted move.
    """
    score = _Budget(score, cap)
    steps = []
    current: tuple = ()
    capped = False
    try:
        best = score(current, prompts)
        steps.append({"move": "start", "rows": current, **best})
        # Greedy add: keep the best clean add while it helps.
        candidates = list(ranked_layers[:top_k])
        while len(current) < max_layers:
            used = {L for L, _ in current}
            trial_best = None
            for L in candidates:
                if L in used:
                    continue
                rows = tuple(sorted(current + ((L, add_dose),)))
                s = score(rows, prompts)
                if fluency_ok(s) and (trial_best is None
                                      or s["gain"] > trial_best[1]["gain"]):
                    trial_best = (rows, s)
            if (trial_best is None
                    or trial_best[1]["gain"] < best["gain"] + min_improve):
                break
            current, best = trial_best
            steps.append({"move": "add", "rows": current, **best})
        # Dose tune: two hill-climbing passes over the mix.
        for _ in range(2):
            changed = False
            for i, (L, d0) in enumerate(current):
                for d in dose_grid:
                    if d == d0:
                        continue
                    rows = current[:i] + ((L, d),) + current[i + 1:]
                    s = score(rows, prompts)
                    if fluency_ok(s) and s["gain"] > best["gain"]:
                        current, best = rows, s
                        steps.append({"move": "dose", "rows": current, **best})
                        changed = True
                        break
            if not changed:
                break
    except _CapHit:
        capped = True
        steps.append({"move": "cap", "rows": current})
    return {"rows": current, "search": best if current else None,
            "steps": steps, "calls": score.calls, "capped": capped}


def prompts_sha256(prompts) -> str:
    """Short hash that names one prompt set in a confirm record."""
    blob = json.dumps(list(prompts), ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def confirm(score, rows, fresh_prompts):
    """Score the winner once on fresh prompts; apply the pass gates.

    The verdict records the spent set (size and hash). A second
    confirm on the same set shows the same hash: re-rolls stay
    visible in the artifact, they never pass as a first try.
    """
    spent = {"fresh_prompts": len(fresh_prompts),
             "prompts_sha256": prompts_sha256(fresh_prompts)}
    if not rows:
        return {"passed": False, "reason": "empty mix", "score": None,
                **spent}
    s = score(rows, fresh_prompts)
    passed = s["gain"] >= MIN_GAIN and fluency_ok(s)
    return {"passed": passed, "score": s, **spent,
            "reason": "pass" if passed else "gain or fluency under the bar"}
