"""Create job: staged trait runs with Stop, status and reattach.

One job runs trait_eval.run_trait in a thread through the stages
extract -> tune -> read_gate -> final -> baseline -> verdict. Jobs
serialize on the engine lock (one model in memory); a second job
stays queued until the first releases it. Stop sets the cancel event;
run_trait raises Stopped at generation granularity.

State persists to jobs/<id>/job.json on every stage/progress update,
so a client can reattach after disconnect; the trait report lands in
jobs/<id>/trait/. Only validated families (trait_store) may start;
anything else raises UnsupportedTrait ("not supported yet").
"""
from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import prompt_bank  # noqa: E402
import trait_eval  # noqa: E402
from trait_store import TraitStore, UnsupportedTrait  # noqa: E402

STAGES = ("extract", "tune", "read_gate", "final", "baseline", "verdict")

DEFAULT_JOBS_DIR = ROOT / "jobs"


@dataclass
class _Job:
    id: str
    trait: str
    seed: int
    model_id: str
    revision: str | None
    research_mode: bool
    status: str = "queued"
    stage: str = STAGES[0]
    done: int = 0
    total: int = 0
    verdict: str | None = None
    frozen: dict = field(default_factory=dict)
    eligibility: dict = field(default_factory=dict)
    guard: dict = field(default_factory=dict)
    error: str = ""
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    cancel: threading.Event = field(default_factory=threading.Event,
                                    repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "trait": self.trait, "seed": self.seed,
            "model_id": self.model_id, "revision": self.revision,
            "research_mode": self.research_mode, "status": self.status,
            "stage": self.stage,
            "progress": {"stage": self.stage, "done": self.done,
                         "total": self.total},
            "verdict": self.verdict, "frozen": self.frozen,
            "eligibility": self.eligibility, "guard": self.guard,
            "error": self.error, "created": self.created,
            "updated": self.updated,
        }


def _default_engine_factory(model_id, revision):
    import engine as engine_mod
    eng = engine_mod.Engine()
    eng.load(model_id, revision)
    return eng


class CreateRunner:
    """Owns Create jobs; at most one touches the model at a time."""

    def __init__(self, jobs_dir: Path | None = None,
                 store: TraitStore | None = None,
                 engine=None, engine_factory=None,
                 run_trait_fn=None):
        self.jobs_dir = Path(jobs_dir) if jobs_dir else DEFAULT_JOBS_DIR
        self.store = store if store is not None else TraitStore()
        self._engine = engine
        self._factory = engine_factory or _default_engine_factory
        self._own_engine = engine is None and engine_factory is None
        self._run_trait = run_trait_fn or trait_eval.run_trait
        self._jobs: dict[str, _Job] = {}
        self._mu = threading.Lock()
        self._local_lock = threading.Lock()

    # -- job control --

    def start(self, trait: str, *, seed: int = 0, model_id: str,
              revision: str | None = None,
              research_mode: bool = False) -> str:
        fam = self.store.resolve(trait)
        if not fam.hand_validated:
            raise UnsupportedTrait(
                f"'{trait}' is not supported yet: family '{fam.name}' "
                f"has no validated judge. Supported families: "
                f"{', '.join(f.name for f in self.store.validated()) or 'none'}.")
        job_id = (f"{fam.name}-{seed}-"
                  f"{time.strftime('%Y%m%d-%H%M%S')}-"
                  f"{len(self._jobs):03d}")
        job = _Job(id=job_id, trait=fam.name, seed=seed,
                   model_id=model_id, revision=revision,
                   research_mode=research_mode)
        with self._mu:
            if job_id in self._jobs:
                raise ValueError(f"job {job_id} already exists")
            self._jobs[job_id] = job
        self._persist(job)
        job.thread = threading.Thread(target=self._run, args=(job_id,),
                                      daemon=True)
        job.thread.start()
        return job_id

    def stop(self, job_id: str) -> dict:
        job = self._get(job_id)
        if job.status in ("done", "stopped", "failed"):
            return job.to_dict()
        job.cancel.set()
        job.thread.join(timeout=120)
        return self.status(job_id)

    def status(self, job_id: str) -> dict:
        return self._get(job_id).to_dict()

    def reattach(self, job_id: str) -> dict:
        """Status from disk; works across processes and restarts."""
        path = self.jobs_dir / job_id / "job.json"
        if not path.exists():
            raise KeyError(f"unknown job {job_id}")
        disk = json.loads(path.read_text())
        with self._mu:
            live = self._jobs.get(job_id)
        if live is not None and live.status in ("queued", "running"):
            return live.to_dict()
        return disk

    def list_jobs(self) -> list[dict]:
        out = []
        if self.jobs_dir.exists():
            for path in sorted(self.jobs_dir.iterdir()):
                job_json = path / "job.json"
                if job_json.exists():
                    out.append(json.loads(job_json.read_text()))
        return out

    # -- internals --

    def _get(self, job_id: str) -> _Job:
        with self._mu:
            if job_id not in self._jobs:
                raise KeyError(f"unknown job {job_id}")
            return self._jobs[job_id]

    def _persist(self, job: _Job) -> None:
        job.updated = time.time()
        d = self.jobs_dir / job.id
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(json.dumps(job.to_dict(), indent=2))

    def _engine_lock(self, eng):
        return getattr(eng, "lock", None) or self._local_lock

    def _run(self, job_id: str) -> None:
        job = self._get(job_id)
        eng = self._engine
        try:
            if eng is None:
                eng = self._factory(job.model_id, job.revision)
            with self._engine_lock(eng):
                if job.cancel.is_set():
                    job.status = "stopped"
                    return
                job.status = "running"
                self._persist(job)

                def progress(stage, done, total):
                    job.stage = stage
                    job.done, job.total = done, total
                    self._persist(job)

                out = self._run_trait(
                    eng, eng.spec, job.trait, prompt_bank.PROMPT_BANK,
                    job.seed, self.jobs_dir / job_id / "trait",
                    job.model_id, getattr(eng, "revision", job.revision),
                    cancel=job.cancel, progress=progress,
                    research_mode=job.research_mode)
                job.stage = "verdict"
                job.verdict = out.get("verdict")
                job.frozen = out.get("frozen") or {}
                job.eligibility = out.get("eligibility") or {}
                job.guard = out.get("guard") or {}
                job.status = "done"
        except trait_eval.Stopped:
            job.status = "stopped"
        except Exception:  # noqa: BLE001 - recorded verbatim for the client
            job.status = "failed"
            job.error = traceback.format_exc()[-4000:]
        finally:
            try:
                if self._own_engine and eng is not None:
                    eng.unload()
            finally:
                self._persist(job)
