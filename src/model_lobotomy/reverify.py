"""Re-run a patch's shipped evidence, as a library (RV).

Same check as scripts/verify_patch.py: base vs steered
generations on held-out final-split prompts, same seed, same
judge, same interval rule. The script keeps working by calling
this module. No printing, no exiting, no behavior change.
"""
from __future__ import annotations

import statistics
import tempfile
from pathlib import Path

from model_lobotomy import engine as engine_mod
from model_lobotomy import evaluation as E
from model_lobotomy import patch as patch_mod
from model_lobotomy import prompt_bank
from model_lobotomy import trait_eval

QUICK_PROMPTS = 16


def final_prompts(seed: int, n: int | None = None) -> list[str]:
    """Held-out final-split prompts for a recipe seed (deterministic)."""
    final = E.make_splits(prompt_bank.PROMPT_BANK, seed).final
    return list(final) if n is None else list(final[:n])


def measure_gains(eng, judge_fn, prompts, edits, seed: int,
                  max_new: int | None = None, progress=None,
                  cancel=None) -> list[float]:
    """Per-prompt steered-minus-base judge gains (base vs steered text)."""
    max_new = trait_eval.GEN_TOKENS if max_new is None else max_new
    gains = []
    total = len(prompts)
    for i, q in enumerate(prompts):
        trait_eval._check(cancel)
        base = eng.run(q, engine_mod.EMPTY_EDITS, medians={},
                       max_new_tokens=max_new, seed=seed)["text"]
        st = eng.run(q, edits, medians={},
                     max_new_tokens=max_new, seed=seed)["text"]
        gains.append(judge_fn(q, st) - judge_fn(q, base))
        if progress is not None:
            progress(i + 1, total)
    return gains


def compare(shipped_card: dict, gains: list[float]) -> dict:
    """Shipped vs measured effect. Pass = measured mean is inside the
    shipped 95% interval."""
    lo, hi = shipped_card["trait_ci"]
    measured = statistics.fmean(gains)
    return {"shipped_mean": shipped_card["trait_mean"],
            "shipped_ci": [lo, hi], "measured_mean": measured,
            "n": len(gains), "pass": bool(lo <= measured <= hi)}


def load_engine(model_id: str):
    """Load the named model on the shared Engine (monkeypatch point)."""
    eng = engine_mod.get_engine()
    eng.load(model_id)
    return eng


def judge_label(meta: dict) -> str | None:
    """The judge label a patch was measured with, or None.

    Order: the patch's own recipe (patches from Create carry it, so
    a third party needs no local family), the local trait store
    (open families), then the legacy spike table.
    """
    label = (meta.get("recipe") or {}).get("judge")
    if isinstance(label, str) and label.strip():
        return label
    trait = meta.get("trait")
    try:
        return trait_eval.family_cfg(trait)["judge"]
    except (KeyError, TypeError):
        pass
    legacy = trait_eval.spike.BEHAVIORS.get(trait)
    return legacy["judge"] if legacy else None


def judge_fn_for(eng, trait: str, label: str | None = None):
    """The trait's frozen judge (monkeypatch point)."""
    if label is None:
        label = trait_eval.spike.BEHAVIORS[trait]["judge"]
    spec = eng.spec
    from model_lobotomy import judges as _judges
    port = _judges.get_judge("local", score_one=lambda q, r, lab: trait_eval.judge_score(
        eng.model, eng.tokenizer, spec, q, r, lab))
    return lambda q, r: port.score_many([(q, r)], label)[0]


def run_verify_patch(patch, *, quick: bool, progress=None,
                     cancel=None, engine=None,
                     load_engine_fn=None,
                     judge_fn_for_fn=None) -> tuple[int, dict]:
    """Run the check. Returns (exit_code, report dict).

    patch is a path or raw .lobot bytes. engine, when given, is
    used as-is (the caller owns its lifetime); else the patch's
    model loads on the shared engine and unloads after.
    """
    tmp = None
    if isinstance(patch, (bytes, bytearray)):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "upload.lobot"
        path.write_bytes(bytes(patch))
    else:
        path = patch
    try:
        loaded = patch_mod.load_patch(path)
    except patch_mod.PatchRejected as e:
        return (2, {"error": f"rejected: {e}"})
    finally:
        if tmp is not None:
            tmp.cleanup()
    meta = loaded["meta"]
    model_id = (meta.get("fingerprint") or {}).get("model_id")
    if not model_id:
        return (2, {"error": "patch names no model_id in fingerprint"})
    seed = (meta.get("recipe") or {}).get("seed", 0)
    trait = meta.get("trait")
    label = judge_label(meta)
    if label is None:
        return (2, {"error": f"unknown trait {trait!r}: the patch "
                    f"carries no judge label and no local family or "
                    f"legacy trait has one"})
    card = meta.get("report_card") or {}
    if "trait_mean" not in card or "trait_ci" not in card:
        return (2, {"error": "patch report_card has no shipped numbers "
                    "(trait_mean, trait_ci)"})
    if meta.get("scale") is None:
        return (2, {"error": "patch has no scale; cannot set strength"})
    unit = tuple(loaded["direction"].detach().cpu().tolist())
    edits = engine_mod.EditSet(vector=(engine_mod.VectorEdit(
        int(meta["layer"]), unit, float(meta["scale"]),
        float(meta["dose"])),))
    prompts = final_prompts(int(seed),
                            QUICK_PROMPTS if quick else None)
    owned = engine is None
    eng = engine if engine is not None else (
        load_engine_fn or load_engine)(model_id)
    try:
        judge_fn = (judge_fn_for_fn or judge_fn_for)(eng, trait, label)
        gains = measure_gains(eng, judge_fn, prompts, edits,
                              int(seed), progress=progress,
                              cancel=cancel)
    finally:
        if owned:
            unload = getattr(eng, "unload", None)
            if callable(unload):
                unload()
    result = compare(card, gains)
    result.update({"trait": trait, "layer": int(meta["layer"]),
                   "dose": float(meta["dose"]), "seed": int(seed),
                   "model_id": model_id, "quick": quick})
    return (0 if result["pass"] else 1, result)
