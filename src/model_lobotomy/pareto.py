"""PP: effect vs side effects per dose, next to the prompt baseline.

For one Create result, every point (vector at several doses, the top
baseline instruction, and the instruction plus the vector) is scored
on the final split with the Create lane's judge on three labels:
the family trait, relevance and fluency. The trait axis is the gain
over the base reply; the cost axis is the mean relevance and fluency
drop. The verdict rule is fixed in docs/roadmap/specs/PP-prompt-pareto.md.

The Pareto is descriptive: the frozen dose stays as tuned.
"""
from __future__ import annotations

import statistics

from model_lobotomy import engine as engine_mod
from model_lobotomy import evaluation as E

PARETO_DOSES = (0.5, 1.0, 1.5, 2.0, 2.5)
# Chosen on real weights (Llama 3.2 3B, 40 base replies, own question vs
# a shifted one; logs/pp_relevance_labels.json): "relevant to the
# question" scored 0.91 vs 0.13 (97.5% pairwise). "a direct answer to
# the question" separated (95%) but scored own replies only 0.11.
RELEVANCE_LABEL = "relevant to the question"
FLUENCY_LABEL = "written in fluent, coherent language"
PARETO_VERSION = "pp-v2"
# The label prompt: a plain instruction built from the judge label.
# Some drafted baselines are sample lines, not instructions, so the
# top baseline alone can be a weak prompt.
LABEL_INSTRUCTION = ("Answer the question. Your reply must match this "
                     "description: {label}")


def doses_for(frozen_dose: float, grid=PARETO_DOSES) -> list[float]:
    """The dose grid plus the frozen dose, sorted, no duplicates."""
    return sorted({float(d) for d in grid} | {float(frozen_dose)})


def _hm(a: float, b: float, c: float) -> float:
    """Harmonic mean of three scores in [0, 1] (0 when any is 0)."""
    if min(a, b, c) <= 0:
        return 0.0
    return 3.0 / (1.0 / a + 1.0 / b + 1.0 / c)


def point_rows(base: dict, scored: dict) -> list[dict]:
    """Per-prompt rows for one point.

    base and scored hold equal-length lists "trait", "relevance",
    "fluency" (raw judge scores). trait is the gain over base; cost
    is the mean of the relevance and fluency drops.
    """
    rows = []
    for i in range(len(scored["trait"])):
        t, r, f = (scored["trait"][i], scored["relevance"][i],
                   scored["fluency"][i])
        rows.append({
            "trait": t - base["trait"][i],
            "cost": ((base["relevance"][i] - r)
                     + (base["fluency"][i] - f)) / 2.0,
            "relevance": r, "fluency": f,
            "hm": _hm(t, r, f)})
    return rows


def summarize(name: str, rows: list[dict], **extra) -> dict:
    """Mean trait gain, cost and hm for one point."""
    return {"name": name, "n": len(rows),
            "trait": statistics.fmean(r["trait"] for r in rows),
            "cost": statistics.fmean(r["cost"] for r in rows),
            "relevance": statistics.fmean(r["relevance"] for r in rows),
            "fluency": statistics.fmean(r["fluency"] for r in rows),
            "hm": statistics.fmean(r["hm"] for r in rows),
            **extra}


def beats(a: dict, b: dict) -> bool:
    """a beats b: trait at least b's and cost at most b's."""
    return a["trait"] >= b["trait"] and a["cost"] <= b["cost"]


def front(points: list[dict]) -> list[str]:
    """Names of points no other point strictly dominates."""
    out = []
    for p in points:
        if not any(beats(q, p) and (q["trait"] > p["trait"]
                                    or q["cost"] < p["cost"])
                   for q in points if q is not p):
            out.append(p["name"])
    return out


def verdict(vector_points: list[dict], prompt_point: dict) -> str:
    """vector_dominates | prompt_dominates | tradeoff (spec rule)."""
    if any(beats(v, prompt_point) for v in vector_points):
        return "vector_dominates"
    if vector_points and all(beats(prompt_point, v)
                             for v in vector_points):
        return "prompt_dominates"
    return "tradeoff"


def strongest_prompt(points: list[dict]) -> dict:
    """The prompt point with the highest trait gain (ties: first)."""
    prompts = [p for p in points if p["kind"] == "prompt"]
    return max(prompts, key=lambda p: p["trait"])


def paired_diff(rows_a: list[dict], rows_b: list[dict], key: str,
                seed: int = 0) -> dict:
    """Mean and bootstrap 95% CI of a[key] - b[key], prompt by prompt."""
    diffs = [a[key] - b[key] for a, b in zip(rows_a, rows_b)]
    lo, hi = E.bootstrap_ci(diffs, n=2000, seed=seed)
    return {"mean": statistics.fmean(diffs), "ci": [lo, hi]}


def _score(lane, prompts, replies, trait_label) -> dict:
    pairs = list(zip(prompts, replies))
    return {"trait": lane.judge_many(pairs, trait_label),
            "relevance": lane.judge_many(pairs, RELEVANCE_LABEL),
            "fluency": lane.judge_many(pairs, FLUENCY_LABEL)}


def run_pareto(lane, label: str, final, *, layer: int, unit, scale,
               frozen_dose: float, instruction: str, max_new: int,
               doses=None, base_cache=None, cancel=None,
               progress=None) -> dict:
    """Score every point on the final split; returns the PP block.

    lane: a Create lane (gen_many, judge_many). unit: the frozen
    direction (tensor); scale: the layer median norm. instruction:
    the top baseline instruction.
    """
    final = list(final)
    doses = doses_for(frozen_dose) if doses is None else list(doses)
    total = (len(doses) + 4) * len(final)
    done = 0

    def _tick():
        nonlocal done
        done += len(final)
        if progress is not None:
            progress("pareto", done, total)

    def _vec(d):
        return engine_mod.EditSet(vector=(engine_mod.VectorEdit(
            layer, tuple(unit.tolist()), float(scale), float(d)),))

    if base_cache is not None:
        bases = base_cache.base_many(lane, final, max_new, cancel=cancel)
    else:
        bases = lane.gen_many(final, engine_mod.EMPTY_EDITS, max_new,
                              cancel=cancel)
    _tick()
    base = _score(lane, final, bases, label)
    prefixed = [f"{instruction}\n\n{q}" for q in final]
    runs = [(f"vector@{d:g}", final, _vec(d), {"kind": "vector",
                                                "dose": d})
            for d in doses]
    runs.append(("prompt", prefixed, engine_mod.EMPTY_EDITS,
                 {"kind": "prompt", "dose": 0.0,
                  "instruction": instruction}))
    label_ins = LABEL_INSTRUCTION.format(label=label)
    runs.append(("prompt:label", [f"{label_ins}\n\n{q}" for q in final],
                 engine_mod.EMPTY_EDITS,
                 {"kind": "prompt", "dose": 0.0,
                  "instruction": label_ins}))
    runs.append(("prompt+vector", prefixed, _vec(frozen_dose),
                 {"kind": "prompt+vector", "dose": float(frozen_dose)}))
    points, rows_by, samples = [], {}, {}
    for name, prompts, edits, extra in runs:
        replies = lane.gen_many(prompts, edits, max_new, cancel=cancel)
        _tick()
        rows = point_rows(base, _score(lane, final, replies, label))
        rows_by[name] = rows
        points.append(summarize(name, rows, **extra))
        samples[name] = [{"q": q, "reply": r[:300]}
                         for q, r in zip(final[:2], replies[:2])]
    vecs = [p for p in points if p["kind"] == "vector"]
    prompt_pt = strongest_prompt(points)
    best = max(vecs, key=lambda p: p["hm"])
    return {
        "version": PARETO_VERSION,
        "labels": {"trait": label, "relevance": RELEVANCE_LABEL,
                   "fluency": FLUENCY_LABEL},
        "layer": layer, "frozen_dose": float(frozen_dose),
        "instruction": instruction, "n": len(final),
        "base": {"trait": statistics.fmean(base["trait"]),
                 "relevance": statistics.fmean(base["relevance"]),
                 "fluency": statistics.fmean(base["fluency"])},
        "points": points,
        "front": front(points),
        "verdict": verdict(vecs, prompt_pt),
        "verdict_prompt": prompt_pt["name"],
        "best_vector": {
            "name": best["name"],
            "trait_minus_prompt": paired_diff(
                rows_by[best["name"]], rows_by[prompt_pt["name"]], "trait"),
            "cost_minus_prompt": paired_diff(
                rows_by[best["name"]], rows_by[prompt_pt["name"]], "cost")},
        "samples": samples,
    }


def pareto_lines(block: dict) -> list[str]:
    """Plain table lines for report.md."""
    lines = [f"Pareto ({block['verdict']} vs "
             f"{block.get('verdict_prompt', 'prompt')}), n={block['n']}, "
             f"layer {block['layer']}:",
             "| point | trait gain | cost | relevance | fluency | hm |",
             "|---|---|---|---|---|---|"]
    for p in block["points"]:
        mark = " *" if p["name"] in block["front"] else ""
        lines.append(
            f"| {p['name']}{mark} | {p['trait']:+.3f} | "
            f"{p['cost']:+.3f} | {p['relevance']:.3f} | "
            f"{p['fluency']:.3f} | {p['hm']:.3f} |")
    return lines
