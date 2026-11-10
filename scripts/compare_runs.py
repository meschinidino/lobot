#!/usr/bin/env python3
"""Compare two Create reports. Refuses a headline when not comparable.

Usage:
  uv run python scripts/compare_runs.py OLD/report.json NEW/report.json

Exit 0: comparable and eligible, deltas printed. Exit 2: not
comparable, a run is headline-ineligible, the workloads differ,
or quality regressed under a speed claim. The reasons print and
no winner is named.

Derived metrics are always recomputed from the stored raw gains
and pairs; a report's own metrics block is never trusted. Only
the lane work counters are carried over, because they are raw
observations that cannot be recomputed after the fact.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_lobotomy import metrics as M  # noqa: E402


def _fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _recomputed(report: dict) -> dict:
    block = M.build_metrics(report, elapsed=report.get("seconds") or 0.0)
    stored = report.get("metrics") or {}
    if stored.get("work"):
        block["work"] = stored["work"]
    return block


def _components(report: dict, verb: str):
    final = report.get("final") or {}
    gains = (final.get(verb) or {}).get("gains") or []
    control = {"update_up": final.get("control_gains"),
               "update_down": final.get("control_removal_gains"),
               "delete": final.get("control_delete_gains")}.get(verb)
    trait_mean = statistics.fmean(gains) if gains else None
    control_mean = (statistics.fmean(control)
                    if control and len(control) == len(gains) else None)
    return trait_mean, control_mean


def compare(a: dict, b: dict) -> tuple[int, list[str]]:
    lines: list[str] = []
    verdict = M.compare_reports(a, b)
    if not verdict["comparable"]:
        lines.append("not comparable: differing "
                     + ", ".join(verdict["differing"]))
        return 2, lines
    ca, cb = M.comparability(a), M.comparability(b)
    problems = []
    if not ca["quality_headline_eligible"]:
        problems.append("old run quality-ineligible: "
                        + ", ".join(ca["quality_reasons"]))
    if not cb["quality_headline_eligible"]:
        problems.append("new run quality-ineligible: "
                        + ", ".join(cb["quality_reasons"]))
    if problems:
        return 2, problems
    ma, mb = _recomputed(a), _recomputed(b)
    paired_means = {}
    for verb in M.VERBS:
        pa = (ma.get("verbs") or {}).get(verb, {}).get("paired")
        pb = (mb.get("verbs") or {}).get(verb, {}).get("paired")
        ta, contro_a = _components(a, verb)
        tb, contro_b = _components(b, verb)
        if pa and pb:
            paired_means[verb] = (pa["mean"], pb["mean"])
            lines.append(
                f"{verb}: trait {_fmt(ta)} -> {_fmt(tb)}; "
                f"control {_fmt(contro_a)} -> {_fmt(contro_b)}; "
                f"paired {_fmt(pa['mean'])} "
                f"[{_fmt(pa['ci'][0])}, {_fmt(pa['ci'][1])}] -> "
                f"{_fmt(pb['mean'])} "
                f"[{_fmt(pb['ci'][0])}, {_fmt(pb['ci'][1])}] "
                f"(n {pa['n']} -> {pb['n']})")
            if (tb is not None and ta is not None and tb <= ta
                    and pb["mean"] > pa["mean"]):
                lines.append(
                    f"  warning: {verb} paired gain rose while the "
                    "trait gain did not; the control got worse, "
                    "not the edit better")
    # Speed: only at fixed quality and matched physical work.
    speed_problems = []
    if not ca["speed_headline_eligible"]:
        speed_problems.append("old run speed-ineligible: "
                              + ", ".join(ca["speed_reasons"]))
    if not cb["speed_headline_eligible"]:
        speed_problems.append("new run speed-ineligible: "
                              + ", ".join(cb["speed_reasons"]))
    wa, wb = ma.get("work") or {}, mb.get("work") or {}
    for field in ("gen_prompts", "score_pairs", "nll_pairs"):
        if wa.get(field) is None or wb.get(field) is None:
            speed_problems.append(f"missing work counter: {field}")
        elif wa[field] != wb[field]:
            speed_problems.append(
                f"workload differs: {field} {wa[field]} != {wb[field]}")
    up = paired_means.get("update_up")
    if up and up[1] < up[0]:
        speed_problems.append(
            "quality regressed on update_up paired effect; "
            "no speed claim at lower quality")
    sa, sb = a.get("seconds"), b.get("seconds")
    if speed_problems:
        lines += speed_problems
        return 2, lines
    lines.append(
        f"seconds (compute stages only, load excluded): "
        f"{_fmt(sa, 1)} -> {_fmt(sb, 1)}; accounted share "
        f"{_fmt(ma['time']['accounted_share'])} -> "
        f"{_fmt(mb['time']['accounted_share'])}")
    lines.append(
        f"work (matched): gen_prompts {wa['gen_prompts']} = "
        f"{wb['gen_prompts']}; score_pairs {wa['score_pairs']} = "
        f"{wb['score_pairs']}; new_tokens "
        f"{wa.get('new_tokens', 'n/a')} -> {wb.get('new_tokens', 'n/a')}")
    if wa.get("new_tokens") and wb.get("new_tokens") and sa and sb:
        lines.append(
            "diagnostic only (rewards verbosity): seconds per 1k "
            f"generated tokens {_fmt(sa / wa['new_tokens'] * 1000)} "
            f"-> {_fmt(sb / wb['new_tokens'] * 1000)}")
    sea = (((ma.get("verbs") or {}).get("update_up") or {})
           .get("side_effects") or {})
    seb = (((mb.get("verbs") or {}).get("update_up") or {})
           .get("side_effects") or {})
    if sea and seb:
        lines.append(
            "update_up side-effect diagnostics (steered): refusal "
            f"{_fmt((sea.get('refusal') or {}).get('steered_rate'))} -> "
            f"{_fmt((seb.get('refusal') or {}).get('steered_rate'))}; "
            "empty "
            f"{_fmt(sea['empty']['steered_rate'])} -> "
            f"{_fmt(seb['empty']['steered_rate'])}; repetition "
            f"{_fmt(sea['repetition']['steered_rate'])} -> "
            f"{_fmt(seb['repetition']['steered_rate'])}; mean words "
            f"{_fmt(sea['length_words']['steered_mean'], 1)} -> "
            f"{_fmt(seb['length_words']['steered_mean'], 1)}")
    lines.append(f"verdicts: {a.get('verdict', '?')} -> "
                 f"{b.get('verdict', '?')}")
    return 0, lines


def main(argv) -> int:
    if len(argv) != 3:
        print("usage: compare_runs.py OLD/report.json NEW/report.json")
        return 2
    a = json.loads(Path(argv[1]).read_text())
    b = json.loads(Path(argv[2]).read_text())
    code, lines = compare(a, b)
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
