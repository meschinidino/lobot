"""lobot web server: FastAPI over the engine, serves the web/ frontend.

Same before/after loop as app.py (Gradio), with a custom frontend that
follows docs/DESIGN.md. Engine state (warm model, presets, active model) stays
in app.py so both UIs load and steer the same way.

Endpoints:
    GET  /            web/index.html
    GET  /api/state   models, active model, warm status, dims, preset rows
    POST /api/model   switch the active model (unloads; next run reloads)
    POST /api/run     one before/after run, streamed as NDJSON stage events
    POST /api/stop    stop the active run (after the current token)
    GET  /api/families        trait families + validation state (D12)
    POST /api/create          start a Create job (staged trait run)
    GET  /api/create/{id}     job status (reattach)
    POST /api/create/{id}/stop  stop a Create job
    GET  /api/jobs            list Create jobs
    GET  /api/traits          finished traits with evidence (E1)
    GET  /api/traits/{id}/report(.md)  trait report JSON (Markdown)
    GET  /api/traits/{id}/patch  trait .lobot patch download (E1)
    POST /api/update  vector Update filmstrip, per-column token NDJSON
    POST /api/neurons click-a-word: top neurons at one token (R1, R2 reply)
    GET  /api/neuron/{layer}/{neuron} one neuron card (R2)
    GET  /api/find    neurons that push one word (R2)
    POST /api/lift    steering lift for one neuron (R2)
    GET  /api/features  trained SAEs for the active model (F1)
    GET  /api/feature/{layer}/{index} one SAE feature card (F1)
    POST /api/feature-lift  steering lift for one SAE feature (F1)
    POST /api/drafts  draft a family from words (C3)
    GET  /api/drafts/{id}  draft parts + labeling queue (C3)
    POST /api/drafts/{id}/labels  submit user labels (C3)
    POST /api/drafts/{id}/validate  agreement + registration (C3)
    GET  /api/chat/{id}/transcript  replay spec for a chat (CH)
    POST /api/chat/replay  re-run a transcript, report match (CH)
    POST /api/chat/start  open a chat session (edits persist, E2)
    POST /api/chat         one chat turn, streamed tokens
    GET  /api/chat/{id}    chat history (reattach)

Full request/response shapes: docs/reference/api.md.

Run:
    uv run python server.py                            # loads weights, serves localhost:7860
    MODEL_LOBOTOMY_NO_MODEL=1 uv run python server.py  # UI only; runs return 503
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from typing import Literal
from os.path import commonprefix
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictBool
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool
from starlette.responses import FileResponse, JSONResponse

from model_lobotomy import app as engine
from model_lobotomy import create_job
from model_lobotomy import drafts as drafts_mod
from model_lobotomy import followups as followups_mod
from model_lobotomy import job_import
from model_lobotomy import jobs_store
from model_lobotomy import guard as guard_mod
from model_lobotomy import guard_screens as guard_screens_mod
from model_lobotomy import combine as combine_mod
from model_lobotomy import installers as install_mod
from model_lobotomy import reverify as reverify_mod
from model_lobotomy import prompt_bank
from model_lobotomy import sae as sae_mod
from model_lobotomy import steer
from model_lobotomy import trait_eval
from model_lobotomy import trait_store
from model_lobotomy.engine import EMPTY_EDITS, FEATURES, Ablation, EditSet, Read, VectorEdit, apply_edits, get_engine
from model_lobotomy.llama_backend import UnsupportedError
from model_lobotomy.steer import RecipeRow

try:
    from model_lobotomy import reader
except ImportError:
    reader = None

ROOT = Path(__file__).resolve().parent.parent.parent
WEB = ROOT / "web"
from model_lobotomy.config import get_settings as _get_settings
_SETTINGS = _get_settings()
NO_MODEL = _SETTINGS.no_model
# Replay mode loads no weights: /api/run answers only setups that were
# recorded (web/example.json, web/sweep*.json) and refuses everything else.
REPLAY = _SETTINGS.replay
REPLAY_STAGE_S = _SETTINGS.replay_stage_s
MAX_DOSES = 8  # Update filmstrip columns per request
# Set by /api/stop; cleared by /api/run and /api/update before the stream
# starts, so a Stop that lands before the stream takes the lock still
# stops that run.
_cancel = threading.Event()

# Lazily built Create runner on the shared Engine (one warm model).
_runner: create_job.CreateRunner | None = None


def _get_runner() -> create_job.CreateRunner:
    global _runner
    if _runner is None:
        _runner = create_job.CreateRunner(engine=get_engine())
    return _runner


_followups: followups_mod.FollowupRunner | None = None


def _get_followups() -> followups_mod.FollowupRunner:
    """Pareto and repair jobs for finished Creates (U4); Create's lane."""
    global _followups
    if _followups is None:
        runner = _get_runner()
        _followups = followups_mod.FollowupRunner(
            runner._engine or get_engine(), lane_fn=runner._lane)
    return _followups

api = FastAPI(title="lobot", docs_url=None, redoc_url=None)


class AppState:
    """One home for the server's shared state, on app.state.

    Properties delegate to the module globals, so tests that set
    server._runner (and friends) keep working.
    """
    _MAP = {"cancel": "_cancel", "runner": "_runner",
            "followups": "_followups", "agents": "_agents",
            "agent_lock": "_agent_lock", "chats": "_chats",
            "sae_cache": "_sae_cache", "install_lock": "_install_lock"}
    def __getattr__(self, name):
        try:
            return globals()[self._MAP[name]]
        except KeyError:
            raise AttributeError(name) from None
    def __setattr__(self, name, value):
        if name in self._MAP:
            globals()[self._MAP[name]] = value
        else:
            super().__setattr__(name, value)


api.state.app_state = AppState()


def _close_when_done(gen) -> None:
    from model_lobotomy import streaming as _streaming
    return _streaming.close_when_done(gen)


def _ndjson(gen) -> StreamingResponse:
    """NDJSON response that closes its generator when the client leaves."""
    from model_lobotomy import streaming as _streaming
    return _streaming.ndjson(gen, _cancel)


def _need_torch(eng, feature: str) -> None:
    if not eng.supports(feature):
        raise HTTPException(501, f"{feature} needs LOBOT_BACKEND=torch. "
                                 f"This server runs the {eng.backend} backend.")


@api.exception_handler(UnsupportedError)
async def _unsupported_handler(_req, exc: UnsupportedError):
    """Backstop: a missed torch-only call is a 501, never a 500.

    Torch never raises UnsupportedError, so torch behavior does not
    change. Explicit _need_torch gates stay mandatory: this cannot
    help after a stream has started.
    """
    return JSONResponse(status_code=501, content={"detail": str(exc)})


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _hostname(host: str) -> str:
    """'127.0.0.1:7860' -> '127.0.0.1'; '[::1]:7860' -> '::1'."""
    host = host.strip().lower()
    if host.startswith("["):
        return host[1:host.find("]")]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


class SameOriginOnly:
    """Reject cross-origin POSTs from browsers (D5), DNS rebinding included.

    A browser POST carries Origin. It passes only when Origin equals the
    server's own scheme://host and that host is a loopback name: a rebound
    domain (evil.example resolving to 127.0.0.1) matches itself, so the
    loopback check is what stops it. Requests without Origin (curl, tests)
    pass. Plain ASGI, so streamed runs and Stop pass through untouched.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "POST":
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
            origin = headers.get("origin")
            if origin is not None:
                host = headers.get("host", "")
                own = f"{scope.get('scheme', 'http')}://{host}".lower().rstrip("/")
                if origin.strip().lower().rstrip("/") != own or _hostname(host) not in LOOPBACK_HOSTS:
                    response = JSONResponse({"detail": "Cross-origin POST rejected."}, status_code=403)
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


api.add_middleware(SameOriginOnly)


def _guard_default() -> bool:
    """Default for "guard" in run/update/create (S1 demo default)."""
    from model_lobotomy.config import get_settings
    return get_settings().guard_default


class Row(BaseModel):
    layer: int
    neuron: int
    dose: float = Field(ge=engine.DOSE_MIN, le=engine.DOSE_MAX)


class RunRequest(BaseModel):
    prompt: str
    rows: list[Row]
    guard: bool = Field(default_factory=_guard_default)
    research_mode: bool = False
    dial_trait: str | None = None


class ModelRequest(BaseModel):
    model_id: str


class CreateRequest(BaseModel):
    description: str
    seed: int = 0
    guard: bool = Field(default_factory=_guard_default)
    research_mode: bool = False
    schedule: Literal["fast", "full"] = "full"
    # None: the PP stage runs on the full schedule and not on fast.
    pareto: bool | None = None


class DraftRequest(BaseModel):
    description: str
    seed: int = 0
    # "muse": Muse Spark (Meta model API) writes the parts; "local":
    # the loaded model does (C3 default).
    drafter: Literal["local", "muse"] = "local"


class AgentRequest(BaseModel):
    description: str
    lane: Literal["torch", "llama"] = "torch"
    max_rounds: int = Field(3, ge=1, le=5)


class LabelRow(BaseModel):
    reply_id: str
    label: StrictBool


class LabelsRequest(BaseModel):
    labels: list[LabelRow] = Field(min_length=1)


class UpdateRequest(BaseModel):
    prompt: str
    trait: str | None = None
    layer: int | None = None
    direction: list[float] | None = None
    scale: float | None = None
    doses: list[float]
    max_new_tokens: int = 60
    guard: bool = Field(default_factory=_guard_default)
    research_mode: bool = False


class NeuronsRequest(BaseModel):
    prompt: str
    rows: list[Row] = []
    trait: str | None = None
    dose: float | None = None
    token_index: int | None = None
    reply_char: int | None = None
    top_k: int = 8
    window: int = 4
    guard: bool = False
    research_mode: bool = False
    reply: str | None = None


class LiftRequest(BaseModel):
    layer: int
    neuron: int
    dose: float = Field(ge=engine.DOSE_MIN, le=engine.DOSE_MAX)
    words: list[str] = Field(min_length=1)
    prompts: list[str] | None = None


class FeatureLiftRequest(BaseModel):
    layer: int
    index: int
    dose: float = Field(ge=engine.DOSE_MIN, le=engine.DOSE_MAX)
    words: list[str] | None = None
    prompts: list[str] | None = None


class DeleteRequest(BaseModel):
    prompt: str
    trait: str
    max_new_tokens: int = 150
    guard: bool = False
    research_mode: bool = False


class InstallRequest(BaseModel):
    trait_id: str = ""
    target: str = ""
    model_ref: str = ""


class CombineCheckRequest(BaseModel):
    patch_ids: list = []
    max_new_tokens: int = 24
    seed: int = 0


class VerifyRequest(BaseModel):
    trait_id: str = ""
    quick: bool = True


class InstallVerifyRequest(BaseModel):
    prompt: str = ""
    judge: str | None = None
    max_new_tokens: int = 64
    # > 0: judge the first N prompts of the patch's own final split
    # (where its shipped interval came from) and compare their mean
    # gain; needs judge "local". The prompt field is then unused.
    judged_n: int = 0


# A judged verify never needs more than the final split (30 prompts).
INSTALL_JUDGED_MAX = 30


class DeleteCheckRequest(BaseModel):
    trait: str
    prompts: list[str] | None = None


class ChatStartRequest(BaseModel):
    rows: list[Row] = []
    trait: str | None = None
    dose: float | None = None
    max_new_tokens: int = 150
    guard: bool = False
    research_mode: bool = False
    dial_traits: list[str] = []


class ReplayTurn(BaseModel):
    message: str
    rows: list[Row] = []


class ReplayRequest(BaseModel):
    transcript: dict


class ChatRequest(BaseModel):
    session_id: str
    message: str
    rows: list[Row] | None = None
    max_new_tokens: int | None = None


def _state() -> dict:
    spec = engine.active_spec()
    preset = engine.active_preset()
    eng = get_engine()
    return {
        "active_model": engine._active_model_id,
        "warm": engine._model is not None and engine._model_id == engine._active_model_id,
        "backend": eng.backend,
        "torch_only": sorted(f for f in FEATURES if not eng.supports(f)),
        "no_model": NO_MODEL,
        "replay": REPLAY,
        "models": [{"id": mid, "short": steer.model_short_name(mid),
                    "gated": "meta-llama" in mid} for mid in sorted(engine._PRESETS)],
        "layers": spec.layers,
        "neurons": spec.intermediate,
        "hidden": spec.hidden,
        "dose_min": engine.DOSE_MIN,
        "dose_max": engine.DOSE_MAX,
        "max_rows": engine.MAX_ROWS,
        "max_new_tokens": int(preset["decode"]["max_new_tokens"]),
        "preset_rows": list(preset["rows"]),
        "guard_default": _guard_default(),
        "guard_screens": list(guard_screens_mod.enabled_screens()),
    }


def _recordings() -> list[dict]:
    """Every recorded run as a /api/run result: example.json, then each web/sweep*.json."""
    out = []
    ex_path = WEB / "example.json"
    if ex_path.exists():
        ex = json.loads(ex_path.read_text())
        rec = {k: ex[k] for k in ("model_id", "prompt", "rows", "baseline", "steered",
                                  "split", "seconds", "recipe", "post")}
        rec["dial"] = ex.get("dial")
        out.append(rec)
    for sw_path in sorted(WEB.glob("sweep*.json")):
        sw = json.loads(sw_path.read_text())
        for r in sw["runs"]:
            out.append({"model_id": sw["model_id"], "prompt": sw["prompt"], "rows": r["rows"],
                        "baseline": sw["baseline"],
                        "steered": {"text": r["text"], "tokens": r["tokens"]},
                        "split": r["split"], "seconds": r["seconds"],
                        "recipe": r.get("recipe", ""), "post": r.get("post", ""),
                        "dial": r.get("dial")})
    return out


def _same_rows(a: list[dict], b: list[dict]) -> bool:
    key = lambda rows: sorted((r["layer"], r["neuron"], round(float(r["dose"]), 4)) for r in rows)
    return key(a) == key(b)


def _find_recording(model_id: str, prompt: str, rows: list[dict]) -> dict | None:
    if rows and all(r["dose"] == 0 for r in rows):
        # Dose 0 changes nothing: any recording of this prompt holds the answer, as its baseline.
        for rec in _recordings():
            if rec["model_id"] == model_id and rec["prompt"].strip() == prompt:
                return {**rec, "rows": rows, "steered": rec["baseline"], "split": None}
        return None
    for rec in _recordings():
        if rec["model_id"] == model_id and rec["prompt"].strip() == prompt and _same_rows(rec["rows"], rows):
            return rec
    return None


def _replay_stream(rec: dict):
    """Same stage events as _run_stream, from a recording, with Stop. Holds the Engine lock."""
    lock = get_engine().lock
    if not lock.acquire(blocking=False):
        yield _event(stage="error", message="A run is in progress. Wait for it to finish.")
        return
    try:
        for stage in ("weights", "baseline", "steered"):
            yield _event(stage=stage)
            if _cancel.wait(REPLAY_STAGE_S):
                yield _event(stage="stopped", after=stage)
                return
        yield _event(stage="complete", result={**rec, "dial": rec.get("dial"),
                                              "cache_hit": True, "replay": True})
    finally:
        lock.release()


def _split_char(tok, text: str, ids: list[int], div: int, other: str) -> int:
    """Char offset of token `div` in `text`; char common prefix as fallback."""
    if div > 0:
        prefix = tok.decode(ids[:div], skip_special_tokens=True)
        if prefix and text.startswith(prefix):
            return len(prefix)
    return len(commonprefix([text, other]))




















_JOB_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _trait_job_dir(job_id: str) -> Path:
    """jobs/<id>/ for a safe id, else 404. Never leaves jobs/."""
    runner = _get_runner()
    if not _JOB_ID_RE.fullmatch(job_id or ""):
        raise HTTPException(404, f"Unknown trait {job_id}.")
    cand = jobs_store.JobPaths(job_id, runner.jobs_dir).root.resolve()
    try:
        cand.relative_to(runner.jobs_dir.resolve())
    except (OSError, ValueError):
        raise HTTPException(404, f"Unknown trait {job_id}.") from None
    if not jobs_store.JobPaths(job_id, runner.jobs_dir).job_json.is_file():
        raise HTTPException(404, f"Unknown trait {job_id}.")
    return jobs_store.JobPaths(job_id, runner.jobs_dir).root


def _card_summary(card: dict | None) -> dict | None:
    """Contract summary of one report card (passed, mean, CI)."""
    if card is None:
        return None
    return {"passed": card["passed"], "trait_mean": card["trait_mean"],
            "trait_ci": list(card["trait_ci"])}


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _repairs(trait_dir: Path) -> dict:
    """Every repair record of a run, by method (gr-v1 is "project")."""
    out = {}
    for method in followups_mod.REPAIR_METHODS:
        rec = _read_json(jobs_store.WorkPaths(trait_dir).repair(method))
        if rec is not None:
            out[method] = rec
    return out


def _chosen_cell(rep: dict) -> dict | None:
    """The tuning cell (layer, dose) that Create froze, with its scores."""
    frozen, table = rep.get("frozen") or {}, (rep.get("tuning") or {}).get("table") or {}
    if frozen.get("layer") is None or frozen.get("dose") is None:
        return None
    row = table.get(str(frozen["layer"])) or {}
    for d, cell in row.items():
        if abs(float(d) - float(frozen["dose"])) < 1e-9:
            return {"layer": frozen["layer"], "dose": frozen["dose"], **cell}
    return None


def _point(pareto: dict | None, dose) -> dict | None:
    if not pareto or dose is None:
        return None
    return next((p for p in pareto.get("points", [])
                 if p.get("kind") == "vector"
                 and abs(float(p.get("dose", -1)) - float(dose)) < 1e-9), None)


def _screen_states(guard: dict | None) -> dict:
    """Per-screen state key: blocked, would_block or passed."""
    if not guard:
        return {}
    screens = guard.get("screens") or {"refusal": guard}
    return {name: ("blocked" if v.get("blocked") else "would_block"
                   if v.get("would_block") else "passed")
            for name, v in screens.items()}


def _enrich_report(trait_dir: Path, rep: dict) -> dict:
    """report.json plus the follow-up results that live beside it (U4)."""
    rep = dict(rep)
    if not rep.get("pareto"):
        rep["pareto"] = _read_json(jobs_store.WorkPaths(trait_dir).pareto)
    rep["repair"] = _repairs(trait_dir) or None
    tuning = dict(rep.get("tuning") or {})
    tuning["chosen"] = _chosen_cell(rep)
    rep["tuning"] = tuning
    return rep


def _evidence_summary(trait_dir: Path, rep: dict) -> dict:
    """List-row summary of the evidence beyond the cards (U4)."""
    rep = _enrich_report(trait_dir, rep)
    pareto, chosen = rep.get("pareto"), rep["tuning"]["chosen"]
    point = _point(pareto, (rep.get("frozen") or {}).get("dose"))
    rule = (rep["tuning"].get("rule") or {}).get("name")
    repairs = rep.get("repair") or {}
    return {
        "pareto_verdict": (pareto or {}).get("verdict"),
        "dose_rule": rule,
        "fluency": (point or {}).get("fluency",
                                     (chosen or {}).get("fluency")),
        "screens": _screen_states(rep.get("guard")),
        "repair": ({m: r.get("status") for m, r in repairs.items()}
                   or None),
    }


def _trait_items() -> list[dict]:
    """One list item per done job with a readable report, newest first.

    version counts finished runs per (family, model), oldest first:
    the first finished run is v1, the next is v2. Jobs without a
    readable report.json are skipped: they hold no evidence to serve.
    """
    runner = _get_runner()
    items = []
    for job in runner.list_jobs():
        if job.get("status") != "done":
            continue
        trait_dir = jobs_store.JobPaths(job["id"], runner.jobs_dir).trait.workdir
        try:
            rep = json.loads(jobs_store.WorkPaths(trait_dir).report.read_text())
        except (OSError, ValueError):
            continue
        frozen = rep.get("frozen") or {}
        fin = rep.get("final") or {}
        base = rep.get("prompt_baseline") or {}
        guard = rep.get("guard")
        items.append({
            "id": job["id"],
            "family": rep.get("trait", job.get("trait")),
            "model_id": job.get("model_id"),
            "version": 0,
            "verdict": rep.get("verdict"),
            "layer": frozen.get("layer"),
            "dose": frozen.get("dose"),
            "created": job.get("created"),
            "cards": {
                "update_up": _card_summary(
                    (fin.get("update_up") or {}).get("card")),
                "update_down": _card_summary(
                    (fin.get("update_down") or {}).get("card")),
                "delete": _card_summary(
                    (fin.get("delete") or {}).get("card")),
                "prompt_baseline": _card_summary(base.get("card")),
            },
            "guard": ({"blocked": guard["blocked"],
                       "would_block": guard["would_block"],
                       "verified": guard["verified"]}
                      if guard is not None else None),
            "patch": (f"/api/traits/{job['id']}/patch"
                      if jobs_store.WorkPaths(trait_dir).patch.is_file() else None),
            "delete_eligible": bool(frozen) and (
                jobs_store.WorkPaths(trait_dir).direction.is_file()),
            "evidence": _evidence_summary(trait_dir, rep),
        })
    items.sort(key=lambda t: (t["created"] or 0, t["id"]))
    seen: dict = {}
    for item in items:
        key = (item["family"], item["model_id"])
        seen[key] = seen.get(key, 0) + 1
        item["version"] = seen[key]
    items.sort(key=lambda t: (t["created"] or 0, t["id"]), reverse=True)
    return items


_DRAFT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{0,60}$")


def _draft_or_404(draft_id: str):
    if not _DRAFT_ID_RE.match(draft_id or ""):
        raise HTTPException(404, f"Unknown draft '{draft_id}'.")
    try:
        return drafts_mod.load_draft(ROOT / "store", draft_id)
    except KeyError:
        raise HTTPException(404, f"Unknown draft '{draft_id}'.") from None




def _muse():
    """A Muse Spark client, or 503 naming what is missing."""
    from model_lobotomy import muse as muse_mod
    try:
        return muse_mod.Muse()
    except muse_mod.MuseError as e:
        raise HTTPException(503, f"Muse Spark is not available: {e}. "
                                 f"Set META_API_KEY.") from None


def _muse_call(fn, *args):
    from model_lobotomy import muse as muse_mod
    try:
        return fn(*args)
    except muse_mod.MuseError as e:
        raise HTTPException(502, f"Muse Spark failed: {e}") from None






_agents: dict[str, dict] = {}
_agent_lock = threading.Lock()




_AGENT_ID_RE = re.compile(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}")


def _agents_dir() -> Path:
    return jobs_store.JobPaths("agents", _get_runner().jobs_dir).root


def _agent_job_id(result: dict) -> str:
    return f"{result.get('family') or 'agent'}-agent-{result['agent_id']}"


def _register_agent_job(result: dict | None) -> str | None:
    """Make the agent's Create run a job the app lists (shelf, evidence).

    The agent writes its run to jobs/agents/<id>/round-<n>; the app
    lists jobs/<id>/job.json only. Returns the job id, or None when the
    agent made no report.
    """
    workdir = Path((result or {}).get("workdir") or "")
    if not result or not jobs_store.WorkPaths(workdir).report.is_file():
        return None
    job_id = _agent_job_id(result)
    jobs_dir = _get_runner().jobs_dir
    if not jobs_store.JobPaths(job_id, jobs_dir).job_json.is_file():
        job_import.import_run(workdir, jobs_dir, job_id)
    return job_id


def _disk_agent(agent_id: str) -> dict | None:
    if not _AGENT_ID_RE.fullmatch(agent_id or ""):
        return None
    rec = _read_json(_agents_dir() / agent_id / "agent.json")
    if rec is None:
        return None
    job_id = _agent_job_id(rec)
    return {"id": agent_id, "status": rec.get("status"),
            "steps": rec.get("steps", []), "result": rec, "error": None,
            "job_id": job_id if jobs_store.JobPaths(job_id, _get_runner().jobs_dir).job_json.is_file() else None}


















class FollowupRequest(BaseModel):
    method: Literal["gradient", "project"] = "gradient"


def _followup_start(job_id: str, kind: str, opts: dict) -> dict:
    trait_dir = jobs_store.JobPaths(job_id, _trait_job_dir(job_id).parent).trait.workdir
    if NO_MODEL or REPLAY:
        raise HTTPException(503, f"{kind.capitalize()} needs model weights. "
                                 "This server runs without them "
                                 "(replay or no-model mode).")
    runner = _get_runner()
    if any(j.get("status") in ("queued", "running")
           for j in runner.list_jobs()):
        raise HTTPException(409, "A Create job is running. "
                                 "Wait for it to finish.")
    job = _read_json(_trait_job_dir(job_id) / "job.json") or {}
    try:
        return _get_followups().start(
            job_id, kind, trait_dir, job.get("model_id"),
            job.get("revision"), **opts)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from None


def _followup_status(job_id: str, kind: str) -> dict:
    trait_dir = jobs_store.JobPaths(job_id, _trait_job_dir(job_id).parent).trait.workdir
    st = _get_followups().status(job_id, kind, trait_dir)
    rep_path = jobs_store.WorkPaths(trait_dir).report
    rep = _read_json(rep_path) if rep_path.is_file() else {}
    if kind == "pareto":
        result = rep.get("pareto") or _read_json(jobs_store.WorkPaths(trait_dir).pareto)
    else:
        result = _repairs(trait_dir) or None
    return {"status": st, "result": result}




















def _delete_edits(knob: dict):
    """Ablation EditSet from a saved knob (DL)."""
    return EditSet(ablate=(Ablation(
        int(knob["layer"]), tuple(knob["direction"])),))




def _delete_stream(prompt: str, trait: str, knob: dict, max_new: int,
                   guard: bool = False, research_mode: bool = False):
    """Off/on ablation filmstrip: guard, then two streamed columns.

    Columns run sequentially (off, then on); each streams its
    tokens as they decode. The on-column carries dial values when
    the trait resolves a dial spec. Holds the Engine lock
    throughout; /api/stop ends the stream after the current token.
    """
    eng = get_engine()
    if not eng.lock.acquire(blocking=False):
        yield _event(stage="error", message="A run is in progress. Wait for it to finish.")
        return
    try:
        layer = int(knob["layer"])
        t_all = time.perf_counter()
        yield _event(stage="weights")
        if eng.model is None or eng.model_id != engine._active_model_id:
            eng.load(engine._active_model_id)
        medians, cal_info = _calibrate(eng, [layer])
        if _cancel.is_set():
            yield _event(stage="stopped", after="weights")
            return
        edits = _delete_edits(knob)
        yield _event(stage="guard")
        gv = _verified_guard(
            eng, edits, medians,
            guard_mod.should_enforce(guard, research_mode))
        if gv.blocked:
            yield _event(stage="blocked", guard=_guard_event_dict(gv))
            return
        if _cancel.is_set():
            yield _event(stage="stopped", after="guard")
            return
        try:
            dial_spec = _dial_spec(eng, trait)
        except KeyError:
            dial_spec = None
        yield _event(stage="columns", modes=["off", "on"],
                     guard=_guard_event_dict(gv))
        texts, counts = ["", ""], [0, 0]
        for col, sets in ((0, EMPTY_EDITS), (1, edits)):
            for ev in eng.run_stream(
                    prompt, sets, medians=medians,
                    max_new_tokens=max_new, seed=engine.SEED,
                    cancel=_cancel,
                    dial=dial_spec if col == 1 else None):
                if ev["type"] == "token":
                    texts[col] += ev["text"]
                    tok_ev: dict = {"col": col, "text": ev["text"],
                                    "mode": "off" if col == 0 else "on"}
                    if ev.get("dial") is not None:
                        tok_ev["dial"] = ev["dial"]
                    yield _event(stage="token", **tok_ev)
                else:
                    counts[col] = ev["result"]["new_tokens"]
            if _cancel.is_set():
                yield _event(stage="stopped", after="columns")
                return
        yield _event(stage="complete", result={
            "model_id": engine._active_model_id, "prompt": prompt,
            "trait": trait, "layer": layer,
            "columns": [{"mode": "off", "text": texts[0],
                         "tokens": counts[0]},
                        {"mode": "on", "text": texts[1],
                         "tokens": counts[1]}],
            "guard": _guard_event_dict(gv),
            "seconds": round(time.perf_counter() - t_all, 1),
            "cache_hit": cal_info["cache_hit"],
        })
    except Exception as e:  # noqa: BLE001 — surface engine errors in the UI
        yield _event(stage="error", message=f"Delete failed: {e}")
    finally:
        eng.lock.release()




_install_lock = threading.Lock()
_INSTALL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")








def _final_prompts(record: dict, n: int) -> list[str]:
    """First n final-split prompts of the install's Create job."""
    report = jobs_store.JobPaths(record["trait_id"], _trait_job_dir(record["trait_id"]).parent).report
    try:
        final = json.loads(report.read_text())["splits"]["final"]
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(422, "The Create report has no final "
                                 "split to judge on.") from None
    return [str(q) for q in final][:n]


def _install_verify_judged(install_id: str, record: dict,
                           req: InstallVerifyRequest) -> dict:
    """Clean vs patched runs on the final split, judged as a mean (IN).

    The shipped interval is a mean over the final split, so the
    measured value is a mean over the same prompts, not one draw.
    """
    prompts = _final_prompts(record, req.judged_n)
    if not _install_lock.acquire(blocking=False):
        raise HTTPException(409, "An install or verify is running. "
                                 "Wait for it to finish.")
    try:
        t_all = time.perf_counter()
        max_new = min(max(int(req.max_new_tokens), 1), 256)
        triples = []
        for q in prompts:
            runs = []
            for patched in (False, True):
                r = install_mod.generate(record["target"], record["run"],
                                         q, patched=patched,
                                         max_new=max_new)
                if not r["ok"]:
                    side = "Patched" if patched else "Unpatched"
                    raise HTTPException(422, f"{side} run failed: "
                                             f"{r['detail']}")
                runs.append(r["text"])
            triples.append((q, runs[0], runs[1]))
        out = {"install_id": install_id, "prompts": prompts,
               "differs": sum(b != p for _q, b, p in triples),
               "comparison": _install_compare_many(record, triples),
               "seconds": round(time.perf_counter() - t_all, 1)}
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (install_mod.INSTALLS_ROOT / install_id /
         f"verify-judged-{stamp}.json").write_text(json.dumps(
             {**out, "texts": [{"q": q, "base": b, "patched": p}
                               for q, b, p in triples]}, indent=2))
        return out
    finally:
        _install_lock.release()


def _install_compare(record: dict, prompt: str, base: str,
                     patched: str) -> dict:
    """Judge gain vs the patch's shipped interval, one prompt (IN)."""
    out = _install_compare_many(record, [(prompt, base, patched)])
    if out.get("per_prompt"):
        one = out.pop("per_prompt")[0]
        out["judge_base"] = one["judge_base"]
        out["judge_patched"] = one["judge_patched"]
    return out


def _install_compare_many(record: dict, triples) -> dict:
    """Mean judge gain over (prompt, base, patched) vs the interval."""
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded "
                                 "(MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to judge.")
    from model_lobotomy import patch as patch_mod
    job_dir = _trait_job_dir(record["trait_id"])
    try:
        meta = patch_mod.load_patch(
            jobs_store.WorkPaths(job_dir / "trait").patch)["meta"]
    except patch_mod.PatchRejected as e:
        raise HTTPException(422, f"patch rejected: {e}") from None
    card = meta.get("report_card") or {}
    if "trait_mean" not in card or "trait_ci" not in card:
        return {"pass": None,
                "reason": "patch has no shipped numbers"}
    trait = meta.get("trait", "")
    try:
        label = trait_eval.family_cfg(trait)["judge"]
    except KeyError:
        raise HTTPException(422, f"no registered family for "
                                 f"{trait!r}") from None
    eng = get_engine()
    if not eng.lock.acquire(blocking=False):
        raise HTTPException(409, "A run is in progress. "
                                 "Wait for it to finish.")
    try:
        if eng.model is None or eng.model_id != engine._active_model_id:
            eng.load(engine._active_model_id)
        spec = engine.active_spec()
        rows = []
        for q, base, patched in triples:
            jb = trait_eval.judge_score(eng.model, eng.tokenizer, spec,
                                        q, base, label)
            js = trait_eval.judge_score(eng.model, eng.tokenizer, spec,
                                        q, patched, label)
            rows.append({"q": q, "judge_base": jb, "judge_patched": js})
    finally:
        eng.lock.release()
    out = install_mod.compare(
        card, [r["judge_patched"] - r["judge_base"] for r in rows])
    out["per_prompt"] = rows
    return out


UPLOAD_BYTES_CAP = 64 * 1024 * 1024


def _verify_pin(patch_path) -> dict:
    """Patch meta, or 422. 409 when its model is not the active one."""
    from model_lobotomy import patch as patch_mod
    try:
        meta = patch_mod.load_patch(patch_path)["meta"]
    except patch_mod.PatchRejected as e:
        raise HTTPException(422, f"patch rejected: {e}") from None
    model_id = (meta.get("fingerprint") or {}).get("model_id")
    if model_id != engine._active_model_id:
        raise HTTPException(
            409, f"Patch model {model_id} is not the active model "
            f"{engine._active_model_id}. Switch models first.")
    return meta






def _verify_stream(patch, quick: bool):
    """NDJSON progress per prompt pair, then one result (RV)."""
    import queue
    eng = get_engine()
    if not eng.lock.acquire(blocking=False):
        yield _event(stage="error", message="A run is in progress. "
                                            "Wait for it to finish.")
        return
    try:
        if eng.model is None or eng.model_id != engine._active_model_id:
            eng.load(engine._active_model_id)
        events: queue.Queue = queue.Queue()

        def progress(done, total):
            events.put(("progress", done, total))

        outcome = {}

        def run():
            try:
                outcome["ret"] = reverify_mod.run_verify_patch(
                    patch, quick=quick, progress=progress,
                    cancel=_cancel, engine=eng)
            except trait_eval.Stopped:
                outcome["stopped"] = True
            except Exception as e:  # noqa: BLE001 — report, not crash
                outcome["error"] = f"Verify failed: {e}"
            finally:
                events.put(("end",))

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        while True:
            item = events.get()
            if item[0] == "end":
                break
            yield _event(stage="progress", done=item[1], total=item[2])
        worker.join()
        if outcome.get("stopped", False) or _cancel.is_set():
            yield _event(stage="stopped", after="verify")
        elif "error" in outcome:
            yield _event(stage="error", message=outcome["error"])
        else:
            code, report = outcome["ret"]
            if "error" in report:
                yield _event(stage="error", message=report["error"])
            else:
                yield _event(stage="done", result=report,
                             exit_code=code)
    finally:
        eng.lock.release()


def _combine_patches(patch_ids: list) -> tuple[list, list]:
    """(names, loaded patches) for trait ids, or 404/422."""
    from model_lobotomy import patch as patch_mod
    if not patch_ids:
        raise HTTPException(400, "Give at least one patch id.")
    if len(patch_ids) > combine_mod.MAX_PATCHES:
        raise HTTPException(
            400, f"Give at most {combine_mod.MAX_PATCHES} patches.")
    names, patches = [], []
    for pid in patch_ids:
        job_dir = _trait_job_dir(pid)
        path = jobs_store.WorkPaths(job_dir / "trait").patch
        if not path.is_file():
            raise HTTPException(422, f"Trait {pid} has no saved "
                                     f"patch to combine.")
        try:
            loaded = patch_mod.load_patch(path)
        except patch_mod.PatchRejected as e:
            raise HTTPException(422, f"patch rejected: {e}") from None
        model_id = (loaded["meta"].get("fingerprint") or {}).get(
            "model_id")
        if model_id != engine._active_model_id:
            raise HTTPException(
                422, f"Patch {pid} targets {model_id}, not the "
                f"active model {engine._active_model_id}.")
        names.append(loaded["meta"].get("trait", pid))
        patches.append(loaded)
    seen: dict[str, int] = {}
    for i, n in enumerate(names):
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[i] = f"{n} #{seen[n]}"
    return names, patches


def _combine_matrix(names, patches, max_new: int, seed: int) -> dict:
    """Pair matrix + install order on the engine (CB)."""
    eng = get_engine()
    if not eng.lock.acquire(blocking=False):
        raise HTTPException(409, "A run is in progress. "
                                 "Wait for it to finish.")
    try:
        if eng.model is None or eng.model_id != engine._active_model_id:
            eng.load(engine._active_model_id)
        layers = sorted({int(p["meta"]["layer"]) for p in patches})
        medians, _ = _calibrate(eng, layers)
        from model_lobotomy import guards as _guards
        guard = _guards.get_guard()
        kw = dict(medians=medians,
                  max_new_tokens=min(max(int(max_new), 1), 64),
                  seed=seed, guard=guard)
        try:
            matrix = combine_mod.pair_matrix(
                names, patches, eng, cancel=_cancel, **kw)
        except trait_eval.Stopped:
            raise HTTPException(409, "Stopped by user.") from None
        by_name = {n: p for n, p in zip(names, patches)}

        def blocked_screens(sub):
            sub = sorted(sub)
            out = combine_mod.check_set(
                sub, [by_name[n] for n in sub], eng,
                cancel=_cancel, **kw)
            return {n for n, s in out["guard"].get(
                "screens", {}).items() if s["blocked"]}

        try:
            order = combine_mod.install_order(names,
                                              blocked_screens)
        except trait_eval.Stopped:
            raise HTTPException(409, "Stopped by user.") from None
        matrix["install_order"] = order
        return matrix
    finally:
        eng.lock.release()






def _trait_vector(knob: dict, dose: float, scale: float):
    """VectorEdit from a saved knob at one dose."""
    return VectorEdit(layer=int(knob["layer"]),
                      direction=tuple(knob["direction"]),
                      scale=float(scale), dose=float(dose))


def _token_ids(tok, wrapped: str) -> list[int]:
    """Wrapped prompt token ids as a plain list (tensor or list in)."""
    out = tok(wrapped, return_tensors="pt")
    ids = out["input_ids"] if isinstance(out, dict) else out.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids]


def _calibrate(eng, layers):
    """(medians, info) for the active model. One helper for every route."""
    preset = engine.active_preset()
    spec = engine.active_spec()
    key = steer.calibration_cache_key(
        engine._active_model_id, eng.revision, eng.tokenizer,
        str(steer.DTYPE), steer.DEVICE)
    with eng.clean():
        try:
            return steer.calibrate_layer_medians(
                eng.model, eng.tokenizer, list(preset["calibration_prompts"]),
                layers or [0],
                cache_path=ROOT / steer.calibration_path_for(
                    engine._active_model_id),
                key=key, spec=spec)
        except UnsupportedError:
            raise UnsupportedError(
                f"no cached medians for layers {layers}; run once "
                f"with LOBOT_BACKEND=torch") from None


def _model_slug(model_id: str) -> str:
    return model_id.lower().replace("/", "-").replace("_", "-")


def _parse_layers(text, spec):
    """'8-27' to an inclusive layer list; None means all layers."""
    if text is None or text == "all":
        return list(range(spec.layers))
    layers = None
    try:
        if "-" in text:
            lo_s, hi_s = text.split("-", 1)
            layers = list(range(int(lo_s), int(hi_s) + 1))
        else:
            layers = [int(text)]
    except (ValueError, AttributeError):
        layers = None
    if not layers or any(not 0 <= layer < spec.layers for layer in layers):
        raise HTTPException(
            400, f"Bad layers {text!r}. Use 'all', '14', or '8-27' "
                 f"inside [0,{spec.layers}).")
    return layers


def _pushes_top(eng, layer: int, neuron: int):
    """Top 3 up tokens for one neuron, or null without the reader."""
    if reader is None:
        return None
    up = reader.pushes_toward(eng, layer, neuron, k=3)["up"]
    return [e["token"] for e in up]


def _rank_neurons(eng, mats, ids, token_index, top_k, window):
    """Top neurons at one row, with top contexts. mats: {L: [T-1, N]}."""
    tok = eng.tokenizer
    row = token_index - 1
    per_layer = []
    for layer, mat in mats.items():
        mat = mat.float()
        vals, idx = torch.topk(mat[row].abs(), top_k)
        per_layer.append((layer, mat, vals, idx))
    cands = []
    for layer, mat, vals, idx in per_layer:
        for v, n in zip(vals.tolist(), idx.tolist()):
            cands.append((v, layer, int(n),
                          float(mat[row, int(n)].item())))
    cands.sort(key=lambda c: -c[0])
    neurons = []
    for _, layer, n, val in cands[:top_k]:
        mat = mats[layer].float()
        col = mat[:, n]
        k = min(3, col.numel())
        _, pos = torch.topk(col, k)
        contexts = []
        for p in pos.tolist():
            t = int(p) + 1
            a, b = max(0, t - window), min(len(ids), t + window + 1)
            contexts.append({
                "position": t,
                "activation": float(col[int(p)].item()),
                "text": tok.decode(ids[a:b], skip_special_tokens=True)})
        neurons.append({"layer": layer, "neuron": n, "activation": val,
                        "top_contexts": contexts,
                        "pushes_toward_top": _pushes_top(eng, layer, n)})
    return neurons


def _reply_token_at(tok, ids, reply_start, char):
    """Index of the reply token that covers char of the reply text.

    Decodes the reply one token longer each step; the first token whose
    decoded end passes char covers it (a space belongs to the next
    word's token, as BPE splits it)."""
    for i in range(reply_start, len(ids)):
        if len(tok.decode(ids[reply_start:i + 1], skip_special_tokens=True)) > char:
            return i
    return len(ids) - 1


def _reply_grabber(grabbed, layer):
    def _pre(_mod, args):
        grabbed[layer] = args[0].detach().float().cpu()
    return _pre


def _capture_reply_mats(eng, edits, medians, ids, spec):
    """{layer: [T-1, N] mlp_in} for raw ids with edits on (reply path).

    capture_with_edits re-wraps text, so it cannot address reply
    tokens. This taps every down_proj input over one forward pass of
    the exact ids and always removes its hooks. Position 0 is dropped
    like reduce="none".
    """
    try:
        device = next(eng.model.parameters()).device
    except StopIteration:
        device = "cpu"
    tensor = torch.tensor([ids]).to(device)
    grabbed = {}
    with apply_edits(eng.model, eng.tokenizer, spec, edits, medians):
        handles = []
        for layer in range(spec.layers):
            mod = steer.resolve_down_proj(eng.model, layer, spec)
            handles.append(mod.register_forward_pre_hook(
                _reply_grabber(grabbed, layer)))
        try:
            with torch.no_grad():
                eng.model(tensor)
        finally:
            for handle in handles:
                handle.remove()
    return {layer: grabbed[layer][0][1:] for layer in range(spec.layers)}










# Loaded SAEs: str(dir) -> (weights mtime, sae, config). Refreshed
# when the weights file changes; tests clear it between cases.
_sae_cache: dict = {}


def _load_sae_cached(sae_dir: Path):
    key = str(sae_dir)
    mtime = (sae_dir / sae_mod.WEIGHTS_NAME).stat().st_mtime
    hit = _sae_cache.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1], hit[2]
    sae, cfg = sae_mod.load_sae(sae_dir)
    _sae_cache[key] = (mtime, sae, cfg)
    return sae, cfg


def _sae_layers() -> list:
    """(layer, dir, config) per trained SAE for the active model."""
    base = ROOT / "store" / "sae" / _model_slug(engine._active_model_id)
    out = []
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if not child.is_dir() or not child.name.startswith("L"):
                continue
            try:
                layer = int(child.name[1:])
            except ValueError:
                continue
            cfg_path = child / sae_mod.CONFIG_NAME
            w_path = child / sae_mod.WEIGHTS_NAME
            if not cfg_path.is_file() or not w_path.is_file():
                continue
            try:
                cfg = json.loads(cfg_path.read_text())
            except (OSError, ValueError):
                continue
            out.append((layer, child, cfg))
    return out








def _chat_edits(sess, rows=None, scale=None):
    """(edits, layers) for a chat turn; rows=None keeps the session's.

    scale=None resolves layers only (the trait scale comes from
    calibration, which needs the layers first).
    """
    use = sess["rows"] if rows is None else rows
    layers = sorted({r.layer for r in use})
    vec = ()
    if sess["trait"] is not None:
        knob = _latest_knob(sess["trait"])
        if knob is None:
            raise KeyError(sess["trait"])
        layers = sorted(set(layers) | {int(knob["layer"])})
        if scale is not None:
            vec = (_trait_vector(knob, sess["dose"], scale),)
    return EditSet(neuron=EditSet.from_rows(use).neuron, vector=vec), layers








def _dials_max_diff(a: dict, b: dict) -> float | None:
    """Max abs value diff over shared dial lists; None when shapes differ."""
    if set(a) != set(b):
        return None
    worst = 0.0
    for trait in a:
        la, lb = a[trait], b[trait]
        if len(la) != len(lb):
            return None
        for ea, eb in zip(la, lb):
            if ea["text"] != eb["text"]:
                return None
            va, vb = ea["value"], eb["value"]
            if (va is None) != (vb is None):
                return None
            if va is not None:
                worst = max(worst, abs(float(va) - float(vb)))
    return worst






def _chat_stream(session_id: str, message: str, rows, max_new: int):
    """One chat turn as token events, then done (edits persist)."""
    sess = _chats[session_id]
    if rows is not None:
        sess["rows"] = rows
    eng = get_engine()
    if not eng.lock.acquire(blocking=False):
        yield _event(stage="error", message="A run is in progress. Wait for it to finish.")
        return
    try:
        t_all = time.perf_counter()
        if eng.model is None or eng.model_id != engine._active_model_id:
            eng.load(engine._active_model_id)
        preset = engine.active_preset()
        spec = engine.active_spec()
        steer.check_preset_binding(preset, engine._active_model_id)
        try:
            _, layers = _chat_edits(sess)
        except KeyError:
            yield _event(stage="error", message=(
                f"No finished Create job for trait '{sess['trait']}' "
                f"with a saved direction. Create the trait first."))
            return
        sess.setdefault("turn_records", [])
        dial_traits = list(sess.get("dial_traits") or [])
        try:
            dial_specs = [_dial_spec(eng, trait)
                          for trait in dial_traits]
        except KeyError as e:
            yield _event(stage="error", message=(
                f"No finished Create job for trait '{e.args[0]}' "
                f"with a saved direction. Create the trait first."))
            return
        layers = sorted(set(layers)
                        | {int(s["layer"]) for s in dial_specs})
        medians, _ = _calibrate(eng, layers)
        scale = None
        if sess["trait"] is not None:
            knob = _latest_knob(sess["trait"])
            scale = float(medians[int(knob["layer"])])
        edits, _ = _chat_edits(sess, scale=scale)
        if dial_specs:
            reads = tuple(Read(layer=int(s["layer"]))
                          for s in dial_specs)
            edits = EditSet(neuron=edits.neuron, vector=edits.vector,
                            ablate=edits.ablate, read=edits.read + reads)
        gv = _verified_guard(
            eng, edits, medians,
            guard_mod.should_enforce(sess["guard"], sess["research_mode"]))
        if gv.blocked:
            yield _event(stage="blocked", guard=_guard_event_dict(gv))
            return
        if _cancel.is_set():
            yield _event(stage="stopped", after="guard")
            return
        messages = list(sess["messages"]) + [{"role": "user",
                                              "content": message}]
        wrapped = _wrap_chat(eng.tokenizer, messages, spec)
        reply = None
        dial_lists: dict[str, list] = {t: [] for t in dial_traits}
        for ev in eng.run_stream(
                message, edits, medians=medians,
                max_new_tokens=max_new, seed=engine.SEED,
                cancel=_cancel, wrapped=wrapped,
                dials=dial_specs or None):
            if ev["type"] == "token":
                tok_ev: dict = {"text": ev["text"]}
                if ev.get("dial") is not None:
                    tok_ev["dial"] = ev["dial"]
                for trait in dial_traits:
                    dial_lists[trait].append(_dial_entry(ev, trait))
                yield _event(stage="token", **tok_ev)
            else:
                reply = ev["result"]
        if reply.get("cancelled"):
            yield _event(stage="stopped", after="chat")
            return
        sess["messages"] = (messages + [{"role": "assistant",
                                         "content": reply["text"]}])
        sess["turn_records"].append({
            "message": message,
            "rows": [{"layer": r.layer, "neuron": r.neuron,
                      "dose": r.dose} for r in sess["rows"]],
            "max_new_tokens": max_new,
            "reply": reply["text"], "dials": dial_lists})
        while len(sess["messages"]) // 2 > MAX_CHAT_TURNS:
            del sess["messages"][:2]
            if sess["turn_records"]:
                del sess["turn_records"][:1]
        yield _event(stage="done", result={
            "session_id": session_id, "turn": len(sess["messages"]) // 2,
            "reply": reply["text"], "tokens": reply["new_tokens"],
            "seconds": round(time.perf_counter() - t_all, 1),
            "dials": dial_lists})
    except Exception as e:  # noqa: BLE001 — surface engine errors in the UI
        yield _event(stage="error", message=f"Chat failed: {e}")
    finally:
        eng.lock.release()


def _latest_knob(trait: str) -> dict | None:
    """Newest finished Create job for a trait with a saved direction, on the active model."""
    runner = _get_runner()
    cands = [j for j in runner.list_jobs()
             if j.get("trait") == trait and j.get("status") == "done"
             and j.get("frozen")
             and j.get("model_id", engine._active_model_id) == engine._active_model_id]
    cands.sort(key=lambda j: j.get("updated", 0), reverse=True)
    for job in cands:
        path = jobs_store.JobPaths(job["id"], runner.jobs_dir).direction
        if not path.exists():
            continue
        from safetensors.torch import load_file
        direction = load_file(str(path))["direction"].tolist()
        return {"layer": job["frozen"]["layer"],
                "dose": job["frozen"]["dose"],
                "direction": direction, "job_id": job["id"]}
    return None


def _event(**kw) -> bytes:
    return (json.dumps(kw) + "\n").encode()


# Fixed neutrals for dial anchors (0 = neg mean, 1 = pos mean).
_DIAL_NEUTRALS = (
    "What do you think about the weather today?",
    "How do I start running?",
    "What should I name my new boat?",
    "Have you read any good books lately?",
)

# Chat sessions: id -> {messages, rows, trait, dose, dial_traits,
# turn_records, max_new_tokens, guard, research_mode, model_id,
# created}. In memory; a restart clears. turn_records holds one
# {message, rows, reply, dials} per turn for replay (CH).
_chats: dict = {}
MAX_CHAT_SESSIONS = 16
MAX_CHAT_TURNS = 40
MAX_CHAT_DIALS = 4


def _dial_spec(eng, trait: str) -> dict:
    """Read-dial spec for a trait: layer, unit direction, 0/1 anchors.

    Anchors are mean token projections of the frozen direction over
    fixed neutral prompts under the trait's neg (0) and pos (1)
    templates. b0/b1 cache in the job's dial.json by fingerprint.
    Raises KeyError when no finished job or no behavior exists.
    """
    knob = _latest_knob(trait)
    if knob is None:
        raise KeyError(trait)
    try:
        cfg = trait_eval.family_cfg(trait)
    except KeyError:
        raise KeyError(trait) from None
    layer = int(knob["layer"])
    u = torch.tensor(knob["direction"], dtype=torch.float32)
    n = float(u.norm().item())
    if n <= 0:
        raise ValueError(f"trait {trait!r} direction has zero norm")
    u = u / n
    fp = eng.fingerprint()
    runner = _get_runner()
    dpath = jobs_store.JobPaths(knob["job_id"], runner.jobs_dir).trait.workdir / "dial.json"
    if dpath.exists():
        try:
            cached = json.loads(dpath.read_text())
        except (json.JSONDecodeError, OSError):
            cached = None
        if (cached and cached.get("fingerprint") == fp
                and int(cached.get("layer", -1)) == layer):
            return {"trait": trait, "layer": layer,
                    "unit": [float(x) for x in u.tolist()],
                    "b0": float(cached["b0"]), "b1": float(cached["b1"]),
                    "job_id": knob["job_id"]}
    pos = [f"{cfg['pos']}\n\n{q}" for q in _DIAL_NEUTRALS]
    neg = [f"{cfg['neg']}\n\n{q}" for q in _DIAL_NEUTRALS]
    cap = eng.capture(pos + neg, [layer], point="block_out",
                      reduce="mean_1")
    dots = [(cap[layer][i].float() @ u).item()
            for i in range(len(pos) + len(neg))]
    b1 = float(sum(dots[:len(pos)]) / len(pos))
    b0 = float(sum(dots[len(pos):]) / len(neg))
    dpath.write_text(json.dumps(
        {"trait": trait, "layer": layer, "b0": b0, "b1": b1,
         "fingerprint": fp}, indent=2))
    return {"trait": trait, "layer": layer,
            "unit": [float(x) for x in u.tolist()],
            "b0": b0, "b1": b1, "job_id": knob["job_id"]}


def _wrap_chat(tok, messages: list[dict], spec) -> str:
    """Chat-template wrap for a message list (thinking off, like runs)."""
    kwargs: dict = {}
    if spec.thinking_kw:
        kwargs["enable_thinking"] = False
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs)


def _guard_event_dict(gv) -> dict:
    return {**guard_mod.verdict_dict(gv),
            "enforced": not gv.research_mode}


def _dial_entry(ev: dict, trait: str) -> dict:
    """One {"text", "value"} dial list entry; null when the tap lagged."""
    d = ev.get("dial")
    return {"text": ev["text"],
            "value": d.get(trait) if isinstance(d, dict) else None}


def _verified_guard(eng, edits, medians, enforce):
    from model_lobotomy import guards as _guards
    return _guards.get_guard().check_verified(
        eng, edits, medians=medians, max_new_tokens=24, seed=engine.SEED,
        judge_fn=guard_mod.refusal_judge_fn(eng),
        research_mode=not enforce, cancel=_cancel)


def _run_stream(prompt: str, rows: list[RecipeRow], spec,
                guard: bool = False, research_mode: bool = False,
                dial_trait: str | None = None):
    """Stage events, then one result event. Holds the Engine lock throughout.

    The lock is taken here, not in run(): a client that disconnects before
    the first chunk never starts this generator, so nothing leaks.
    The Engine comes from get_engine() so tests inject a fake (same asserts).
    dial_trait streams per-token read-dial values during the steered stage
    and adds one {"text", "value"} per generated token for both passes to
    complete.result.dial ("baseline", "steered").
    """
    eng = get_engine()
    if not eng.lock.acquire(blocking=False):
        yield _event(stage="error", message="A run is in progress. Wait for it to finish.")
        return
    try:
        preset = engine.active_preset()
        calib_prompts = list(preset["calibration_prompts"])
        max_new = int(preset["decode"]["max_new_tokens"])
        t_all = time.perf_counter()

        yield _event(stage="weights")
        if eng.model is None or eng.model_id != engine._active_model_id:
            eng.load(engine._active_model_id)
        steer.check_preset_binding(preset, engine._active_model_id)
        model, tok = eng.model, eng.tokenizer
        dial_spec = None
        if dial_trait is not None:
            try:
                dial_spec = _dial_spec(eng, dial_trait)
            except KeyError:
                yield _event(stage="error", message=(
                    f"No finished Create job for trait '{dial_trait}' "
                    f"with a saved direction. Create the trait first."))
                return
        layers = sorted({r.layer for r in rows}
                        | ({dial_spec["layer"]} if dial_spec else set()))
        medians, cal_info = _calibrate(eng, layers)
        calib_id = cal_info["key"]["prompts_sha256"]
        if _cancel.is_set():
            yield _event(stage="stopped", after="weights")
            return

        yield _event(stage="guard")
        gv = _verified_guard(
            eng, EditSet.from_rows(rows), medians,
            guard_mod.should_enforce(guard, research_mode))
        if gv.blocked:
            yield _event(stage="blocked", guard=_guard_event_dict(gv))
            return
        if _cancel.is_set():
            yield _event(stage="stopped", after="guard")
            return

        yield _event(stage="baseline")
        dial_baseline: list = []
        dial_steered: list = []
        if dial_spec is None:
            base = eng.run(prompt, EMPTY_EDITS, medians=medians,
                           max_new_tokens=max_new, seed=engine.SEED, cancel=_cancel)
        else:
            base = None
            read_only = EditSet(read=(Read(layer=dial_spec["layer"]),))
            for ev in eng.run_stream(
                    prompt, read_only, medians=medians,
                    max_new_tokens=max_new, seed=engine.SEED,
                    cancel=_cancel, dial=dial_spec):
                if ev["type"] == "token":
                    dial_baseline.append(_dial_entry(ev, dial_spec["trait"]))
                else:
                    base = ev["result"]
        if _cancel.is_set():
            yield _event(stage="stopped", after="baseline")
            return

        yield _event(stage="steered")
        if dial_spec is None:
            steered = eng.run(prompt, EditSet.from_rows(rows),
                              medians=medians, max_new_tokens=max_new,
                              seed=engine.SEED, cancel=_cancel)
        else:
            steered = None
            edits = EditSet(neuron=EditSet.from_rows(rows).neuron,
                            read=(Read(layer=dial_spec["layer"]),))
            for ev in eng.run_stream(
                    prompt, edits, medians=medians,
                    max_new_tokens=max_new, seed=engine.SEED,
                    cancel=_cancel, dial=dial_spec):
                if ev["type"] == "token":
                    dial_steered.append(_dial_entry(ev, dial_spec["trait"]))
                    tok_ev: dict = {"text": ev["text"]}
                    if ev.get("dial") is not None:
                        tok_ev["dial"] = ev["dial"]
                    yield _event(stage="token", **tok_ev)
                else:
                    steered = ev["result"]
        hook_calls = steered["hook_calls"]
        if _cancel.is_set():
            yield _event(stage="stopped", after="steered")
            return
        if hook_calls < 1:
            yield _event(stage="error", message="Steering hooks did not fire. "
                                                "The outputs are not trustworthy.")
            return

        div = engine.first_divergence(base["token_ids"], steered["token_ids"])
        split = None
        if div is not None:
            split = {
                "token": div,
                "base_char": _split_char(tok, base["text"], base["token_ids"], div,
                                         steered["text"]),
                "steered_char": _split_char(tok, steered["text"], steered["token_ids"],
                                            div, base["text"]),
            }
        kw = dict(rows=rows, calib_id=calib_id, base=base, steered=steered,
                  max_new_tokens=max_new, seed=engine.SEED,
                  calib_prompts=calib_prompts, model_id=engine._active_model_id)
        yield _event(
            stage="complete",
            result={
                "model_id": engine._active_model_id,
                "prompt": prompt,
                "rows": [{"layer": r.layer, "neuron": r.neuron, "dose": r.dose}
                         for r in rows],
                "baseline": {"text": base["text"], "tokens": base["new_tokens"]},
                "steered": {"text": steered["text"], "tokens": steered["new_tokens"]},
                "split": split,
                "seconds": round(time.perf_counter() - t_all, 1),
                "cache_hit": cal_info["cache_hit"],
                "guard": _guard_event_dict(gv),
                "dial": ({"trait": dial_spec["trait"],
                          "layer": dial_spec["layer"],
                          "neutral": dial_spec["b0"],
                          "trait_mean": dial_spec["b1"],
                          "baseline": dial_baseline,
                          "steered": dial_steered}
                         if dial_spec else None),
                "recipe": steer.recipe_string(
                    rows, calib_id, max_new_tokens=max_new, seed=engine.SEED,
                    calib_prompts=calib_prompts, model_id=engine._active_model_id),
                "post": steer.build_post_text(**kw),
            })
    except Exception as e:  # noqa: BLE001 — surface engine errors in the UI
        yield _event(stage="error", message=f"Run failed: {e}")
    finally:
        eng.lock.release()


def _update_stream(prompt: str, layer: int, direction: list,
                   scale: float | None, doses: list[float],
                   max_new: int, spec, guard: bool = False,
                   research_mode: bool = False):
    """Vector Update filmstrip: guard, then per-column token events.

    One batched generate across doses (E6 hook, singles fallback);
    each column streams its tokens as they decode. Holds the Engine
    lock throughout; /api/stop ends the stream after the current
    token. The guard checks the strongest dose before any column.
    """
    eng = get_engine()
    if not eng.lock.acquire(blocking=False):
        yield _event(stage="error", message="A run is in progress. Wait for it to finish.")
        return
    try:
        preset = engine.active_preset()
        calib_prompts = list(preset["calibration_prompts"])
        t_all = time.perf_counter()

        yield _event(stage="weights")
        if eng.model is None or eng.model_id != engine._active_model_id:
            eng.load(engine._active_model_id)
        steer.check_preset_binding(preset, engine._active_model_id)
        model, tok = eng.model, eng.tokenizer
        medians, cal_info = _calibrate(eng, [layer])
        if scale is None:
            scale = float(medians[layer])
        if _cancel.is_set():
            yield _event(stage="stopped", after="weights")
            return

        unit = tuple(direction)
        strongest = max(doses, key=abs)
        yield _event(stage="guard")
        gv = _verified_guard(
            eng, EditSet(vector=(VectorEdit(layer, unit, scale, strongest),)),
            medians, guard_mod.should_enforce(guard, research_mode))
        if gv.blocked:
            yield _event(stage="blocked", guard=_guard_event_dict(gv))
            return
        if _cancel.is_set():
            yield _event(stage="stopped", after="guard")
            return

        yield _event(stage="columns", doses=doses, scale=scale,
                     guard=_guard_event_dict(gv))
        sets = [EditSet(vector=(VectorEdit(layer, unit, scale, d),))
                for d in doses]
        columns = [""] * len(doses)
        result: dict = {}
        for ev in eng.run_batched_stream(
                prompt, sets, medians=medians, max_new_tokens=max_new,
                seed=engine.SEED, cancel=_cancel):
            if ev["type"] == "token":
                columns[ev["col"]] += ev["text"]
                yield _event(stage="token", col=ev["col"],
                             text=ev["text"], dose=doses[ev["col"]],
                             fallback=bool(ev.get("fallback")))
            else:
                result = ev["result"]
        if _cancel.is_set():
            yield _event(stage="stopped", after="columns")
            return
        yield _event(
            stage="complete",
            result={
                "model_id": engine._active_model_id,
                "prompt": prompt,
                "layer": layer,
                "scale": scale,
                "doses": doses,
                "columns": [{"dose": d, "text": r["text"],
                             "tokens": r["new_tokens"]}
                            for d, r in zip(doses, result["rows"])],
                "fallback_used": result["fallback_used"],
                "guard": _guard_event_dict(gv),
                "seconds": round(time.perf_counter() - t_all, 1),
                "cache_hit": cal_info["cache_hit"],
            })
    except Exception as e:  # noqa: BLE001 — surface engine errors in the UI
        yield _event(stage="error", message=f"Update failed: {e}")
    finally:
        eng.lock.release()


# Last, so /api/* routes win. web/ at the root keeps the page's relative
# paths (vendor/, example.json) the same under any static server.
# -- routers (MOD phase 8) --
from model_lobotomy.routers import (  # noqa: E402
    chat as _r_chat, create as _r_create, drafts_agent as _r_drafts,
    evidence as _r_evidence, install_verify as _r_install,
    reader_features as _r_reader, run as _r_run)
_DYNAMIC = {"NO_MODEL", "REPLAY", "REPLAY_STAGE_S", "ROOT", "WEB",
            "_runner", "_followups", "_cancel", "reader"}
for _r in (_r_run, _r_create, _r_evidence, _r_drafts, _r_chat,
           _r_install, _r_reader):
    # Inject real types so FastAPI resolves annotations. Dynamic
    # names stay out: routers read them live from this module
    # through their module __getattr__ (tests patch them here).
    def _delegate(_name, _orig):
        import types as _types
        if not isinstance(_orig, _types.FunctionType):
            return _orig
        def _wrap(*a, **kw):
            return getattr(sys.modules[__name__], _name)(*a, **kw)
        return _wrap
    _r.__dict__.update({k: _delegate(k, v) for k, v in globals().items()
                        if k not in _DYNAMIC and not k.startswith("__")})
    api.include_router(_r.router)

api.mount("/", StaticFiles(directory=WEB, html=True), name="web")



if __name__ == "__main__":
    if REPLAY:
        print(f"[server] REPLAY set: no weights; {len(_recordings())} recorded runs "
              "answer /api/run", flush=True)
    elif NO_MODEL:
        print("[server] NO_MODEL set: serving the UI without weights", flush=True)
    else:
        engine.load_engine()
        print(f"[server] engine warm ({engine._model_id}); models: "
              f"{sorted(engine._PRESETS)}", flush=True)
    from model_lobotomy.config import get_settings
    uvicorn.run(api, host="127.0.0.1", port=get_settings().port)
