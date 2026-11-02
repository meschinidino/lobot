"""Register a finished run_trait work dir as a Create job for the app.

run_trait writes evidence to any work dir (logs/<run>, or the Create
agent's jobs/agents/<id>/round-<n>). The app lists only
jobs/<id>/job.json. import_run copies the work dir to jobs/<id>/trait/
and writes job.json from the report. Nothing is re-measured.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

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
