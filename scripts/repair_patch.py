"""GR for a finished Create run: repair a patch the guard flags.

Confirms the patch's guard with full generation, and when a screen
flags, tries the GR candidates in their fixed order. Writes
JOBDIR/repair_<method>.json (repair.json for project) and, when a
candidate is chosen, JOBDIR/knob.repaired-<method>.lobot. Rules:
docs/roadmap/specs/GR-guard-repair.md. The logic lives in
followups.repair_job, which the server's repair route also runs.

Run:
    uv run python scripts/repair_patch.py jobs/pirate-0-20261030-004543-000/trait --lane llama
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_lobotomy import followups  # noqa: E402
from model_lobotomy import trait_eval  # noqa: E402
from model_lobotomy.engine import get_engine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobdir")
    ap.add_argument("--lane", default="torch", choices=["torch", "llama"])
    ap.add_argument("--method", default="gradient",
                    choices=list(followups.REPAIR_METHODS),
                    help="gradient: remove the safe replies' NLL gradients "
                         "(gr-v2); project: remove mean-diff safety "
                         "directions, then lower doses (gr-v1)")
    args = ap.parse_args()
    model_id = trait_eval.load_job(args.jobdir)["report"]["fingerprint"][
        "model_id"]
    eng = get_engine()
    eng.load(model_id)

    def log(msg):
        print(f"[repair] {msg}", flush=True)

    out = followups.repair_job(
        eng, args.jobdir, trait_eval.open_lane(eng, args.lane, model_id),
        method=args.method, log=log)
    log(f"status={out['status']} flagged={out.get('flagged')} "
        f"chosen={(out.get('chosen') or {}).get('method')}"
        f"@{(out.get('chosen') or {}).get('factor')}")
    stem = followups.repair_stem(args.method)
    log(f"wrote {Path(args.jobdir) / f'{stem}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
