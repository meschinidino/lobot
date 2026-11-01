"""PP for a finished Create run: effect vs side effects per dose.

Reads JOBDIR/report.json and its knob.lobot, scores every point on
the run's final split, and writes JOBDIR/pareto.json. The verdict
rule is in docs/roadmap/specs/PP-prompt-pareto.md.

Run:
    uv run python scripts/pareto.py logs/c3_cowboy_llama --lane llama
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

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
    job = trait_eval.load_job(args.jobdir)
    report, meta = job["report"], job["meta"]
    model_id = report["fingerprint"]["model_id"]
    cfg = trait_eval.family_cfg(report["trait"])
    max_new = args.gen_tokens or trait_eval.schedule_params(
        report.get("schedule", "full"))["gen_tokens"]
    eng = get_engine()
    eng.load(model_id)
    lane = (trait_eval.open_lane(eng, args.lane, model_id)
            or trait_eval.TorchLane(eng, eng.spec))
    block = pareto.run_pareto(
        lane, cfg["judge"], report["splits"]["final"],
        layer=int(meta["layer"]), unit=job["direction"].float(),
        scale=float(meta["scale"]), frozen_dose=float(meta["dose"]),
        instruction=report["prompt_baseline"]["top"], max_new=max_new,
        progress=lambda s, d, t: print(f"[pareto] {d}/{t}", flush=True))
    block["lane"] = lane.info()
    block["trait"] = report["trait"]
    out = Path(args.jobdir) / "pareto.json"
    out.write_text(json.dumps(block, indent=2))
    print("\n".join(pareto.pareto_lines(block)), flush=True)
    print(f"[pareto] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
