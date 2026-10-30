"""Run one trait end to end (wave-0 E2E proof).

Loads one model on the Engine, runs the full pipeline (derive, tune,
Read gate, final per-verb gains, shuffled control, prompt baseline,
report cards, eligibility, ledger), writes logs/e2e_<trait>/.

Run:
    uv run python scripts/run_trait.py --trait blunt
    uv run python scripts/run_trait.py --trait blunt --model Qwen/Qwen3.5-4B --seed 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import get_engine  # noqa: E402
import prompt_bank  # noqa: E402
import steer  # noqa: E402
import trait_eval  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="blunt")
    ap.add_argument("--model", default=steer.LLAMA_MODEL_ID)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--guard", action="store_true",
                    help="enforce the refusal guard (default: record only)")
    args = ap.parse_args()
    if args.trait not in trait_eval.spike.BEHAVIORS:
        print(f"unknown trait {args.trait!r}; have "
              f"{sorted(trait_eval.spike.BEHAVIORS)}", flush=True)
        return 2
    if args.trait not in trait_eval.BASELINE_INSTRUCTIONS:
        print(f"trait {args.trait!r} has no O4 baseline instructions; "
              f"add it to trait_eval.BASELINE_INSTRUCTIONS first",
              flush=True)
        return 2
    eng = get_engine()
    rung = eng.load(args.model)
    print(f"[trait] {args.model} warm (rung {rung}, rev {eng.revision[:12]})",
          flush=True)
    workdir = Path(args.workdir or (ROOT / "logs" / f"e2e_{args.trait}"))
    out = trait_eval.run_trait(eng, eng.spec, args.trait,
                               prompt_bank.PROMPT_BANK, args.seed,
                               workdir, args.model, eng.revision,
                               guard=args.guard)
    print(f"[trait] verdict={out['verdict']} eligibility={out['eligibility']} "
          f"({out['seconds']} s)", flush=True)
    print(f"[trait] report: {workdir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
