"""Combined-patch checks: pairs and sets, not singles (CB).

Builds one EditSet per patch subset (vectors sum on shared
layers, ablations compose in layer order) and runs one
verified guard check per subset. Union fingerprints are
order-free: vectors sort deterministically, so {A, B} and
{B, A} share a cache entry.
"""
from __future__ import annotations

import math
import time
from itertools import combinations

from model_lobotomy import engine as engine_mod
from model_lobotomy import guard as guard_mod

MAX_PATCHES = 8
# |cosine| above which a same-layer vector/ablation pair warns.
OVERLAP_COSINE = 0.5


def _vec_key(v) -> tuple:
    d = tuple(v.direction)
    return (v.layer, v.dose, v.scale, d[:4])


def edits_for(patches: list[dict]):
    """One EditSet for loaded patch dicts ({direction, meta}).

    Vector patches add (sorted, so order-free); a patch whose
    meta sets edit_kind "ablate" becomes an ablation; neuron
    rows come from meta "rows" when present.
    """
    vectors, ablations, neurons = [], [], []
    for p in patches:
        meta = p["meta"]
        raw = p["direction"]
        direction = tuple(raw.tolist() if hasattr(raw, "tolist")
                          else list(raw))
        if meta.get("edit_kind") == "ablate":
            ablations.append(engine_mod.Ablation(
                int(meta["layer"]), direction))
            continue
        vectors.append(engine_mod.VectorEdit(
            int(meta["layer"]), direction, float(meta["scale"]),
            float(meta["dose"])))
        for row in meta.get("rows") or ():
            neurons.append(engine_mod.NeuronEdit(
                int(row["layer"]), int(row["neuron"]),
                float(row["dose"])))
    vectors.sort(key=_vec_key)
    ablations.sort(key=lambda a: (a.layer, tuple(a.direction)[:4]))
    neurons.sort(key=lambda e: (e.layer, e.neuron, e.dose))
    return engine_mod.EditSet(vector=tuple(vectors),
                              ablate=tuple(ablations),
                              neuron=tuple(neurons))


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def overlap_warnings(edits) -> list[str]:
    """UI strings for vector/ablation same-layer overlap (CB).

    A vector add (L, v) overlaps an ablation (L, u) when their
    |cosine| tops 0.5: the ablation partly removes what the
    knob adds. Neuron/ablation overlap comes from the engine's
    own overlap_warning and joins the list.
    """
    out = []
    by_layer: dict[int, list] = {}
    for a in edits.ablate:
        by_layer.setdefault(a.layer, []).append(a)
    for v in edits.vector:
        for a in by_layer.get(v.layer, []):
            if abs(_cosine(tuple(v.direction),
                           tuple(a.direction))) > OVERLAP_COSINE:
                out.append(
                    f"Vector L{v.layer} pushes along a deleted "
                    f"direction; the ablation partly removes it.")
    neuron_warn = engine_mod.overlap_warning(edits)
    if neuron_warn is not None:
        out.append(neuron_warn)
    return out


def check_set(names, patches, engine, *, medians: dict,
              max_new_tokens: int, seed: int, judge_fn=None,
              cancel=None, guard=None) -> dict:
    """One verified check over a patch subset (CB).

    guard, when given, is shared across subsets so union
    verdicts cache (callers build one per matrix).
    """
    judge_fn = judge_fn or guard_mod.refusal_judge_fn(engine)
    edits = edits_for(patches)
    from model_lobotomy import guards as _guards
    gv = (guard or _guards.get_guard()).check_verified(
        engine, edits, medians=medians,
        max_new_tokens=max_new_tokens, seed=seed,
        judge_fn=judge_fn, cancel=cancel)
    d = guard_mod.verdict_dict(gv)
    return {"patches": list(names), "blocked": d["blocked"],
            "would_block": d["would_block"],
            "verified": d["verified"], "guard": d,
            "overlap": overlap_warnings(edits),
            "cached": gv.cached}


def pair_matrix(names, patches, engine, *, medians: dict,
                max_new_tokens: int, seed: int, judge_fn=None,
                cancel=None, guard=None) -> dict:
    """Singles + all unordered pairs + the full set (CB)."""
    t_all = time.perf_counter()
    kw = dict(medians=medians, max_new_tokens=max_new_tokens,
              seed=seed, judge_fn=judge_fn, cancel=cancel,
              guard=guard or _guards.get_guard())
    singles = [check_set([n], [p], engine, **kw)
               for n, p in zip(names, patches)]
    pairs = [check_set([names[i], names[j]],
                       [patches[i], patches[j]], engine, **kw)
             for i, j in combinations(range(len(names)), 2)]
    full = check_set(list(names), list(patches), engine, **kw)
    checks = len(singles) + len(pairs) + 1
    cached = sum(1 for r in singles + pairs + [full]
                 if r["cached"])
    return {"singles": singles, "pairs": pairs, "full_set": full,
            "checks": checks, "cached": cached,
            "seconds": round(time.perf_counter() - t_all, 1)}


def install_order(names, blocked_screens_fn) -> list[str]:
    """Greedy safest-first install order (CB, deterministic).

    blocked_screens_fn(frozenset_of_names) returns the blocked
    screen names for that subset. Each step installs the
    remaining patch adding the fewest newly blocked screens;
    ties break by name.
    """
    remaining = list(names)
    ordered: list[str] = []
    blocked: set = set()
    while remaining:
        def cost(n):
            got = set(blocked_screens_fn(
                frozenset(ordered + [n])))
            return (len(got - blocked), n)
        nxt = min(remaining, key=cost)
        remaining.remove(nxt)
        ordered.append(nxt)
        blocked |= set(blocked_screens_fn(frozenset(ordered)))
    return ordered
