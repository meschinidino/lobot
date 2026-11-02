"""Import a finished run_trait.py work dir as a Create job for the app.

run_trait.py writes its evidence to any work dir (logs/<run> by
habit). The app lists only jobs/<id>/job.json. This copies the work
dir to jobs/<id>/trait/ and writes job.json from the report, so the
evidence page, Pareto and repair routes can use the run. Nothing is
re-measured.

Run:
    uv run python scripts/import_run.py logs/fd_pirate_llama
    uv run python scripts/import_run.py logs/clane_pirate_llama --id pirate-cli-l17
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_lobotomy.create_job import DEFAULT_JOBS_DIR  # noqa: E402

SKIP = {"job.json"}


def job_record(job_id: str, report: dict, created: float) -> dict:
    """job.json for a finished run, from its report (pure)."""
    fp = report.get("fingerprint") or {}
    return {
        "id": job_id, "trait": report["trait"],
        "seed": report.get("seed", 0),
        "model_id": fp.get("model_id"), "revision": fp.get("revision"),
        "research_mode": False, "guard_opt_in": False,
        "schedule": report.get("schedule", "full"),
        "pareto": bool(report.get("pareto")),
        "status": "done", "stage": "verdict",
        "progress": {"stage": "verdict", "done": 1, "total": 1},
        "verdict": report.get("verdict"),
        "frozen": report.get("frozen") or {},
        "eligibility": report.get("eligibility") or {},
        "guard": report.get("guard") or {},
        "error": "", "created": created,
        "updated": created + float(report.get("seconds") or 0),
        "imported_from": None,
    }


def import_run(src: Path, jobs_dir: Path, job_id: str | None = None,
               force: bool = False) -> Path:
    src = Path(src)
    report = json.loads((src / "report.json").read_text())
    created = (src / "report.json").stat().st_mtime - float(
        report.get("seconds") or 0)
    job_id = job_id or (f"{report['trait']}-{report.get('seed', 0)}-"
                        f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(created))}"
                        f"-cli")
    dest = Path(jobs_dir) / job_id
    if dest.exists():
        if not force:
            raise FileExistsError(f"{dest} exists; pass --force to replace")
        shutil.rmtree(dest)
    shutil.copytree(src, dest / "trait",
                    ignore=lambda d, names: [n for n in names if n in SKIP])
    rec = job_record(job_id, report, created)
    rec["imported_from"] = str(src)
    (dest / "job.json").write_text(json.dumps(rec, indent=2))
    return dest


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
