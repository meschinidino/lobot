"""Import a finished run_trait.py work dir as a Create job for the app.

run_trait.py writes its evidence to any work dir (logs/<run> by
habit). The app lists only jobs/<id>/job.json. This copies the work
dir to jobs/<id>/trait/ and writes job.json from the report, so the
evidence page, Pareto and repair routes can use the run. Nothing is
re-measured. The logic is job_import.import_run.

Run:
    uv run python scripts/import_run.py logs/fd_pirate_llama
    uv run python scripts/import_run.py logs/clane_pirate_llama --id pirate-cli-l17
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_lobotomy.jobs_store import default_jobs_dir as _djd  # noqa: E402
from model_lobotomy.jobs_store import import_run, job_record  # noqa: E402,F401
DEFAULT_JOBS_DIR = _djd()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("--id", default=None, help="job id (default: from the report)")
    ap.add_argument("--jobs-dir", default=str(DEFAULT_JOBS_DIR))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    dest = import_run(Path(args.workdir), Path(args.jobs_dir), args.id,
                      args.force)
    print(f"[import] {args.workdir} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
