"""Job storage: every jobs/<id>/ path in one place (MOD phase 2).

`JobPaths(job_id)` gives the paths of one job. `WorkPaths(dir)` gives
the evidence files inside one trait work dir. `list_jobs()` and
`import_run()` live here; `job_import` re-exports them.
The jobs dir is `LOBOT_JOBS_DIR`, else `<repo>/jobs`.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SKIP = {"job.json"}


def default_jobs_dir() -> Path:
    from model_lobotomy.config import get_settings
    return get_settings().jobs_dir or (ROOT / "jobs")


def repair_stem(method: str) -> str:
    """repair.json for project (gr-v1), repair_<method>.json otherwise."""
    return "repair" if method == "project" else f"repair_{method}"


@dataclass(frozen=True)
class WorkPaths:
    """Evidence files inside one trait work dir."""
    workdir: Path

    @property
    def report(self) -> Path:
        return self.workdir / "report.json"

    @property
    def patch(self) -> Path:
        return self.workdir / "knob.lobot"

    @property
    def direction(self) -> Path:
        return self.workdir / "direction.safetensors"

    @property
    def pareto(self) -> Path:
        return self.workdir / "pareto.json"

    def repair(self, method: str) -> Path:
        return self.workdir / f"{repair_stem(method)}.json"

    def repaired_patch(self, method: str) -> Path:
        return self.workdir / f"knob.repaired-{method}.lobot"


@dataclass(frozen=True)
class JobPaths:
    """Paths of one job under a jobs dir."""
    job_id: str
    jobs_dir: Path | None = None

    @property
    def base(self) -> Path:
        return self.jobs_dir or default_jobs_dir()

    @property
    def root(self) -> Path:
        return self.base / self.job_id

    @property
    def job_json(self) -> Path:
        return self.root / "job.json"

    @property
    def trait(self) -> WorkPaths:
        return WorkPaths(self.root / "trait")

    # Flat aliases used by callers that held a trait dir before.
    @property
    def report(self) -> Path:
        return self.trait.report

    @property
    def patch(self) -> Path:
        return self.trait.patch

    @property
    def direction(self) -> Path:
        return self.trait.direction

    @property
    def pareto(self) -> Path:
        return self.trait.pareto

    def repair(self, method: str) -> Path:
        return self.trait.repair(method)


def list_jobs(jobs_dir=None) -> list[dict]:
    base = Path(jobs_dir) if jobs_dir else default_jobs_dir()
    out = []
    if base.exists():
        for path in sorted(base.iterdir()):
            job_json = path / "job.json"
            if job_json.exists():
                out.append(json.loads(job_json.read_text()))
    return out


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
    """Copy a finished run_trait work dir to jobs/<id>/trait/."""
    src = Path(src)
    report = json.loads((src / "report.json").read_text())
    created = (src / "report.json").stat().st_mtime - float(
        report.get("seconds") or 0)
    job_id = job_id or (f"{report['trait']}-{report.get('seed', 0)}-"
                        f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(created))}"
                        f"-cli")
    paths = JobPaths(job_id, Path(jobs_dir))
    dest = paths.root
    if dest.exists():
        if not force:
            raise FileExistsError(f"{dest} exists; pass --force to replace")
        shutil.rmtree(dest)
    shutil.copytree(src, paths.trait.workdir,
                    ignore=lambda d, names: [n for n in names if n in SKIP])
    rec = job_record(job_id, report, created)
    rec["imported_from"] = str(src)
    paths.job_json.write_text(json.dumps(rec, indent=2))
    return dest
