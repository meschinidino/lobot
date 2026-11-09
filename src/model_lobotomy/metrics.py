"""Objective Create metrics: speed and output quality, hard to game.

One versioned block (`report.metrics`) plus a provenance envelope
(`report.provenance`). Design rules, from the metrics audit
(docs/reference/metrics.md):

- No single scalar score. Speed is read only at fixed quality and
  coverage; quality is read only at a fixed prompt count.
- Every headline number must be recomputable from raw per-prompt
  values stored in the same report (final gains, pairs, timings).
- A run that cannot support a headline says so in
  `comparability.reasons` (fast schedule, fake judge, guard not
  enforced, early stop, small n). The flags never change the
  verdict; they stop a reader from quoting the number as comparable.
- Missing inputs stay missing (None). Nothing is padded with a
  neutral score.
- Side effects here are judge-independent text measures (refusal
  markers, empty, repetition, realized length). One judge measures
  both effect and cost elsewhere; these do not share that failure.

Pure helpers only: no torch, no model calls. Covered by
tests/test_metrics.py.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import statistics
import subprocess  # noqa: S404 - git rev-parse for a provenance stamp

from model_lobotomy import evaluation as E

METRICS_VERSION = "metrics-v1"

# Ports whose presence means the numbers are simulated, not measured.
SIMULATED_PORTS = {"judge": {"fake"}, "guard": {"noop"},
                   "dose_rule": {"fake"}}

# A confirmatory final needs the full split. Fast runs cap at 10.
MIN_FINAL_N = 30


def bank_sha256(prompts) -> str:
    """Content hash of a prompt bank, in bank order.

    Canonical JSON, not a join: ["a\\nb", "c"] and ["a", "b\\nc"]
    must not hash the same.
    """
    canonical = json.dumps(list(prompts), ensure_ascii=False,
                             separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def label_sha256(label: str) -> str:
    """Content hash of a judge label. Editing the label edits the target."""
    return hashlib.sha256(label.encode()).hexdigest()


@functools.lru_cache(maxsize=1)
def git_sha():
    """Short git sha of the checkout, or None outside a git tree.

    A dirty tree gets a `-dirty` suffix: uncommitted code is not
    the code at HEAD, and the stamp must not say it is.
    """
    def _run(args):
        out = subprocess.run(args, capture_output=True, text=True,  # noqa: S603
                             timeout=2, stdin=subprocess.DEVNULL)
        return out.stdout.strip() if out.returncode == 0 else None

    try:
        sha = _run(["git", "rev-parse", "--short", "HEAD"])  # noqa: S607
        if not sha:
            return None
        dirty = _run(["git", "status", "--porcelain"])  # noqa: S607
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{sha}-dirty" if dirty else sha


def paired_effect(trait_gains, control_gains, *, seed: int = 0) -> dict:
    """Paired trait-minus-control effect: the quality headline shape.

    Mean and bootstrap CI of the per-prompt differences, plus the
    win rate and the worst decile. Means alone hide tails; the
    decile and win rate show them. Raises ValueError on mismatched
    or empty inputs.
    """
    trait = list(trait_gains)
    control = list(control_gains)
    if not trait or len(trait) != len(control):
        raise ValueError(
            f"paired gains must be non-empty and equal-length: "
            f"{len(trait)} != {len(control)}")
    if not all(math.isfinite(v) for v in trait + control):
        raise ValueError("paired gains contain non-finite values")
    diffs = [t - c for t, c in zip(trait, control)]
    lo, hi = E.bootstrap_ci(diffs, seed=seed)
    ordered = sorted(diffs)
    p10 = ordered[max(0, int(0.10 * len(ordered)) - 1)]
    return {"n": len(diffs), "mean": statistics.fmean(diffs),
            "ci": [lo, hi],
            "win_rate": sum(1 for d in diffs if d > 0) / len(diffs),
            "p10": p10}


def _words(text: str) -> list[str]:
    return text.split()


def _is_repetitive(text: str) -> bool:
    """Heuristic collapse check: few distinct words in a long reply."""
    ws = _words(text.lower())
    if len(ws) < 8:
        return False
    return len(set(ws)) / len(ws) < 0.5


def text_side_effects(pairs, *, refusal_fn=None) -> dict:
    """Judge-independent side effects on stored final pairs.

    pairs: [{"q", "base", "steered", ...}]. Refusal uses the
    caller's marker gate when given; without one the refusal block
    is None (missing), never a fabricated zero. Length is the
    realized reply length in words: the schedule caps tokens, so
    length itself is a free variable that moves both speed and
    judge scores.
    """
    pairs = list(pairs)
    n = len(pairs)
    if n == 0:
        raise ValueError("text_side_effects needs at least one pair")
    bases = [p["base"] for p in pairs]
    steered = [p["steered"] for p in pairs]

    def rate(texts, fn):
        return sum(1 for t in texts if fn(t)) / n

    refusal = None
    if refusal_fn is not None:
        base_rate = rate(bases, refusal_fn)
        steered_rate = rate(steered, refusal_fn)
        refusal = {"base_rate": base_rate, "steered_rate": steered_rate,
                   "delta": steered_rate - base_rate}
    return {
        "n": n,
        "refusal": refusal,
        "empty": {"base_rate": rate(bases, lambda t: not t.strip()),
                  "steered_rate": rate(steered,
                                       lambda t: not t.strip())},
        "repetition": {"base_rate": rate(bases, _is_repetitive),
                       "steered_rate": rate(steered, _is_repetitive)},
        "length_words": {
            "base_mean": statistics.fmean(len(_words(t)) for t in bases),
            "steered_mean": statistics.fmean(
                len(_words(t)) for t in steered)},
    }


def stage_accounting(elapsed: float, timings: dict) -> dict:
    """Wall time split: staged, unstaged, and the accounted share.

    Stage timings can look fast when work moves outside a timed
    stage. The unaccounted remainder makes that move visible.
    """
    accounted = sum(float(v) for v in timings.values())
    unaccounted = max(0.0, float(elapsed) - accounted)
    share = accounted / elapsed if elapsed > 0 else None
    # Stages cannot take longer than the whole run. A small slack
    # covers rounding; beyond it the timings contradict the total.
    valid = elapsed > 0 and accounted <= elapsed * 1.05 + 0.05
    return {"elapsed_seconds": float(elapsed),
            "accounted_seconds": accounted,
            "unaccounted_seconds": unaccounted,
            "accounted_share": share,
            "valid": bool(valid)}


VERBS = ("update_up", "update_down", "delete")


def comparability(report: dict) -> dict:
    """May this run support a speed or quality headline? If not, why.

    Fails closed: missing evidence is a reason, not a pass.
    Reasons are stable machine strings, split into
    `quality_reasons` and `speed_reasons` (speed adds its own).
    The comparison key names every field two runs must share
    before their numbers may be compared.
    """
    quality: list[str] = []
    prov = report.get("provenance") or {}
    ports = report.get("ports") or {}
    for port, bad in SIMULATED_PORTS.items():
        if ports.get(port) is None:
            quality.append(f"missing_port:{port}")
        elif ports.get(port) in bad:
            quality.append(f"simulated_port:{port}={ports[port]}")
    if report.get("schedule") != "full":
        quality.append("schedule_not_full")
    for field in ("bank_sha256", "gen_tokens", "judge_label_sha256",
                  "schedule_params"):
        if not prov.get(field):
            quality.append(f"missing_provenance:{field}")
    if not report.get("frozen"):
        quality.append("missing_frozen_candidate")
    final = report.get("final") or {}
    split_final = (report.get("splits") or {}).get("final") or []
    for verb in VERBS:
        gains = (final.get(verb) or {}).get("gains")
        if not gains:
            quality.append(f"missing_final_gains:{verb}")
            continue
        if len(gains) < MIN_FINAL_N:
            quality.append("final_n_below_30")
        if split_final and len(gains) != len(split_final):
            quality.append("final_n_mismatch_split")
        if not all(math.isfinite(v) for v in gains):
            quality.append("nonfinite_gains")
        pairs = (final.get(verb) or {}).get("pairs") or []
        if pairs:
            qs = [p.get("q") for p in pairs]
            if len(set(qs)) != len(qs):
                quality.append("duplicate_pairs")
            if split_final and qs != split_final[:len(qs)]:
                quality.append("pair_order_mismatch")
    stops = report.get("early_stop") or {}
    if any(s.get("stopped_early") for s in stops.values()):
        quality.append("stopped_early")
    guard = report.get("guard")
    if not guard:
        quality.append("missing_guard_result")
    else:
        if guard.get("enforced") is not True:
            quality.append("guard_not_enforced")
        if guard.get("blocked"):
            quality.append("guard_blocked")
    quality = sorted(set(quality))
    speed = list(quality)
    if report.get("capture_cache_hit"):
        speed.append("capture_cache_hit")
    elapsed = report.get("seconds")
    if elapsed is not None and report.get("timings"):
        if not stage_accounting(float(elapsed),
                                 report["timings"])["valid"]:
            speed.append("timing_inconsistent")
    judge = report.get("judge") or {}
    key = {"schedule": report.get("schedule"),
           "gen_tokens": prov.get("gen_tokens"),
           "schedule_params": prov.get("schedule_params"),
           "bank_sha256": prov.get("bank_sha256"),
           "judge_label_sha256": prov.get("judge_label_sha256"),
           "fingerprint": report.get("fingerprint"),
           "lane": report.get("lane"),
           "judge": ports.get("judge"),
           "judge_model": judge.get("model"),
           "seed": report.get("seed")}
    return {"quality_headline_eligible": not quality,
            "speed_headline_eligible": not speed,
            "quality_reasons": quality,
            "speed_reasons": sorted(set(speed)),
            "reasons": sorted(set(speed)),
            "comparison_key": key}


def compare_reports(a: dict, b: dict) -> dict:
    """Are two reports comparable at all? Names the differing key fields.

    A key field missing on either side counts as differing: two
    unknowns are not an equality.
    """
    ka = comparability(a)["comparison_key"]
    kb = comparability(b)["comparison_key"]
    differing = sorted(
        k for k in ka
        if ka[k] is None or kb[k] is None or ka[k] != kb[k])
    return {"comparable": not differing, "differing": differing,
            "key_a": ka, "key_b": kb}


def build_metrics(report: dict, *, elapsed: float,
                  refusal_fn=None) -> dict:
    """The metrics block for one Create report.

    Reads only fields already in the report dict (final gains,
    control gains, pairs, frozen, timings), so every number here
    can be recomputed from the stored artifact. Verbs without
    stored control gains keep `paired: None`.
    """
    final = report.get("final") or {}
    controls = {"update_up": final.get("control_gains"),
                "update_down": final.get("control_removal_gains"),
                "delete": final.get("control_delete_gains")}
    verbs = {}
    for verb in ("update_up", "update_down", "delete"):
        block = final.get(verb) or {}
        gains = block.get("gains") or []
        entry = {"n": len(gains), "paired": None, "side_effects": None}
        control = controls.get(verb) or []
        if gains and len(control) == len(gains):
            try:
                entry["paired"] = paired_effect(gains, control)
            except ValueError:
                entry["paired"] = None
        if block.get("pairs"):
            entry["side_effects"] = text_side_effects(
                block["pairs"], refusal_fn=refusal_fn)
        verbs[verb] = entry
    frozen = report.get("frozen") or {}
    up_gains_all = (final.get("update_up") or {}).get("gains") or []
    up_mean = (statistics.fmean(up_gains_all) if up_gains_all
               else ((final.get("update_up") or {}).get("card")
                     or {}).get("trait_mean"))
    tuning_gain = frozen.get("tuning_gain")
    gap = (tuning_gain - up_mean
           if tuning_gain is not None and up_mean is not None else None)
    baseline = report.get("prompt_baseline") or {}
    vector_vs_prompt = None
    up_gains = (final.get("update_up") or {}).get("gains") or []
    base_gains = baseline.get("final_gains") or []
    if up_gains and len(up_gains) == len(base_gains):
        try:
            vector_vs_prompt = paired_effect(up_gains, base_gains,
                                             seed=7)
        except ValueError:
            vector_vs_prompt = None
    return {"version": METRICS_VERSION,
            "verbs": verbs,
            "winners_curse_gap": gap,
            "vector_vs_prompt": vector_vs_prompt,
            "time": stage_accounting(elapsed,
                                      report.get("timings") or {}),
            "comparability": comparability(report)}
