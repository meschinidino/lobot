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

from model_lobotomy.reverify import (  # noqa: E402
    QUICK_PROMPTS, compare, final_prompts, judge_fn_for,
    load_engine, measure_gains, run_verify_patch)

__all__ = ["QUICK_PROMPTS", "compare", "final_prompts",
           "judge_fn_for", "load_engine", "measure_gains",
           "run_verify", "main"]


def run_verify(patch_path, *, quick: bool) -> tuple[int, dict]:
    """Run the check. Returns (exit_code, report dict)."""
    return run_verify_patch(patch_path, quick=quick,
                            load_engine_fn=load_engine,
                            judge_fn_for_fn=judge_fn_for)


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
