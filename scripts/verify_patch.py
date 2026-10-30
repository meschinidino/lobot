"""Re-run a patch's shipped evidence on the named model.

Loads the model named in the patch fingerprint, re-runs base vs
steered generations on held-out final-split prompts (same seed as
the recipe, same judge as the trait), and prints shipped vs
measured effect with PASS or FAIL. --quick uses 16 prompts; the
default run uses the full 30-prompt final split.

Needs the model lane (loads weights).

Run:
    uv run python scripts/verify_patch.py knob.lobot --quick
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine as engine_mod  # noqa: E402
import evaluation as E  # noqa: E402
import patch as patch_mod  # noqa: E402
import prompt_bank  # noqa: E402
import trait_eval  # noqa: E402

QUICK_PROMPTS = 16


def final_prompts(seed: int, n: int | None = None) -> list[str]:
    """Held-out final-split prompts for a recipe seed (deterministic)."""
    final = E.make_splits(prompt_bank.PROMPT_BANK, seed).final
    return list(final) if n is None else list(final[:n])


def measure_gains(eng, judge_fn, prompts, edits, seed: int,
                  max_new: int | None = None) -> list[float]:
    """Per-prompt steered-minus-base judge gains (base vs steered text)."""
    max_new = trait_eval.GEN_TOKENS if max_new is None else max_new
    gains = []
    for q in prompts:
        base = eng.run(q, engine_mod.EMPTY_EDITS, medians={},
                       max_new_tokens=max_new, seed=seed)["text"]
        st = eng.run(q, edits, medians={},
                     max_new_tokens=max_new, seed=seed)["text"]
        gains.append(judge_fn(q, st) - judge_fn(q, base))
    return gains


def compare(shipped_card: dict, gains: list[float]) -> dict:
    """Shipped vs measured effect. Pass = measured mean is inside the
    shipped 95% interval."""
    import statistics
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


def judge_fn_for(eng, trait: str):
    """The trait's frozen judge (monkeypatch point)."""
    label = trait_eval.spike.BEHAVIORS[trait]["judge"]
    spec = eng.spec
    return lambda q, r: trait_eval.judge_score(
        eng.model, eng.tokenizer, spec, q, r, label)


def run_verify(patch_path, *, quick: bool) -> tuple[int, dict]:
    """Run the check. Returns (exit_code, report dict)."""
    try:
        loaded = patch_mod.load_patch(patch_path)
    except patch_mod.PatchRejected as e:
        return (2, {"error": f"rejected: {e}"})
    meta = loaded["meta"]
    model_id = (meta.get("fingerprint") or {}).get("model_id")
    if not model_id:
        return (2, {"error": "patch names no model_id in fingerprint"})
    seed = (meta.get("recipe") or {}).get("seed", 0)
    trait = meta.get("trait")
    if trait not in trait_eval.spike.BEHAVIORS:
        return (2, {"error": f"unknown trait {trait!r}; have "
                    f"{sorted(trait_eval.spike.BEHAVIORS)}"})
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
    eng = load_engine(model_id)
    try:
        judge_fn = judge_fn_for(eng, trait)
        gains = measure_gains(eng, judge_fn, prompts, edits,
                              int(seed))
    finally:
        unload = getattr(eng, "unload", None)
        if callable(unload):
            unload()
    result = compare(card, gains)
    result.update({"trait": trait, "layer": int(meta["layer"]),
                   "dose": float(meta["dose"]), "seed": int(seed),
                   "model_id": model_id, "quick": quick})
    return (0 if result["pass"] else 1, result)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("patch", help=".lobot patch file")
    ap.add_argument("--quick", action="store_true",
                    help=f"check {QUICK_PROMPTS} held-out prompts "
                    f"(default: the full final split)")
    args = ap.parse_args(argv)
    code, out = run_verify(args.patch, quick=args.quick)
    if "error" in out:
        print(f"[verify] {out['error']}", flush=True)
        return code
    print(f"[verify] trait {out['trait']} layer {out['layer']} "
          f"dose {out['dose']} seed {out['seed']} model {out['model_id']}",
          flush=True)
    print(f"[verify] prompts: {out['n']}"
          f"{' (quick)' if out['quick'] else ''}", flush=True)
    lo, hi = out["shipped_ci"]
    print(f"[verify] shipped effect {out['shipped_mean']:.3f} "
          f"(95% CI [{lo:.3f}, {hi:.3f}]); "
          f"measured effect {out['measured_mean']:.3f}", flush=True)
    if out["pass"]:
        print("[verify] PASS: measured is inside the shipped interval",
              flush=True)
    else:
        print("[verify] FAIL: measured is outside the shipped interval",
              flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
