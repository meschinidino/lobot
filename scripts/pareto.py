"""PP for a finished Create run: effect vs side effects per dose.

Reads JOBDIR/report.json and its knob.lobot, scores every point on
the run's final split, and writes JOBDIR/pareto.json. The verdict
rule is in docs/roadmap/specs/PP-prompt-pareto.md. The logic lives in
followups.pareto_job, which the server's Pareto route also runs.

Run:
    uv run python scripts/pareto.py logs/c3_cowboy_llama --lane llama
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_lobotomy import followups  # noqa: E402
from model_lobotomy import pareto  # noqa: E402
from model_lobotomy import trait_eval  # noqa: E402
from model_lobotomy.engine import get_engine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobdir")
    ap.add_argument("--lane", default="torch", choices=["torch", "llama"])
    ap.add_argument("--gen-tokens", type=int, default=None,
                    help="default: the run's schedule")
    args = ap.parse_args()
    model_id = trait_eval.load_job(args.jobdir)["report"]["fingerprint"][
        "model_id"]
    eng = get_engine()
    eng.load(model_id)
    block = followups.pareto_job(
        eng, args.jobdir, trait_eval.open_lane(eng, args.lane, model_id),
        max_new=args.gen_tokens,
        progress=lambda s, d, t: print(f"[pareto] {d}/{t}", flush=True))
    print("\n".join(pareto.pareto_lines(block)), flush=True)
    print(f"[pareto] wrote {Path(args.jobdir) / 'pareto.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
