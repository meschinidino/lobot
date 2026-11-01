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

import json
import os
import re
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
from starlette.responses import FileResponse, JSONResponse

from model_lobotomy import app as engine
from model_lobotomy import create_job
from model_lobotomy import drafts as drafts_mod
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
NO_MODEL = bool(os.environ.get("MODEL_LOBOTOMY_NO_MODEL"))
# Replay mode loads no weights: /api/run answers only setups that were
# recorded (web/example.json, web/sweep*.json) and refuses everything else.
REPLAY = bool(os.environ.get("MODEL_LOBOTOMY_REPLAY"))
REPLAY_STAGE_S = float(os.environ.get("MODEL_LOBOTOMY_REPLAY_STAGE_S", "0.4"))
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

api = FastAPI(title="lobot", docs_url=None, redoc_url=None)


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
    return os.environ.get("LOBOT_GUARD_DEFAULT") == "1"


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


@api.get("/api/state")
def state():
    return _state()


@api.post("/api/model")
def switch_model(req: ModelRequest):
    if req.model_id not in engine._PRESETS:
        raise HTTPException(400, f"Unknown model {req.model_id}. "
                                 f"Supported: {', '.join(sorted(engine._PRESETS))}.")
    if get_engine().lock.locked():
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    engine._active_model_id = req.model_id
    engine.unload_engine()
    return _state()


@api.post("/api/run")
def run(req: RunRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(400, "The prompt is empty. Type a prompt first.")
    if not req.rows:
        raise HTTPException(400, "The recipe is empty. Add at least one row.")
    if len(req.rows) > engine.MAX_ROWS:
        raise HTTPException(400, f"Too many rows. The limit is {engine.MAX_ROWS}.")
    rows = [RecipeRow(layer=r.layer, neuron=r.neuron, dose=r.dose) for r in req.rows]
    spec = engine.active_spec()
    try:
        steer.validate_recipe(rows, spec)
    except ValueError as e:
        raise HTTPException(400, f"Invalid row: {e}.") from None
    if REPLAY:
        rec = _find_recording(engine._active_model_id, prompt, [r.model_dump() for r in req.rows])
        if rec is None:
            raise HTTPException(404, "Replay mode answers only recorded setups, and this one was "
                                     "not recorded. Record it with scripts/record_sweep.py, or "
                                     "restart without MODEL_LOBOTOMY_REPLAY to run the model.")
        if get_engine().lock.locked():
            raise HTTPException(409, "A run is in progress. Wait for it to finish.")
        _cancel.clear()
        return StreamingResponse(_replay_stream(rec), media_type="application/x-ndjson")
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    if req.dial_trait is not None:
        _need_torch(get_engine(), "dial")
    if get_engine().lock.locked():
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    _cancel.clear()
    return StreamingResponse(
        _run_stream(prompt, rows, spec, guard=req.guard,
                    research_mode=req.research_mode,
                    dial_trait=req.dial_trait),
        media_type="application/x-ndjson")


@api.post("/api/stop")
def stop():
    _cancel.set()
    return {"stopping": get_engine().lock.locked()}


@api.get("/api/families")
def families():
    runner = _get_runner()
    return {"families": [
        {"name": f.name, "aliases": list(f.aliases),
         "pos": f.pos, "neg": f.neg, "judge": f.judge,
         "hand_validated": f.hand_validated,
         "validation": f.validation}
        for f in runner.store.all()]}


@api.post("/api/create")
def create(req: CreateRequest):
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    _need_torch(get_engine(), "create")
    runner = _get_runner()
    try:
        job_id = runner.start(req.description, seed=req.seed,
                              model_id=engine._active_model_id,
                              guard=req.guard,
                              research_mode=req.research_mode,
                              schedule=req.schedule)
    except trait_store.UnsupportedTrait as e:
        raise HTTPException(422, str(e)) from None
    return {"job_id": job_id, **runner.status(job_id)}


@api.get("/api/create/{job_id}")
def create_status(job_id: str):
    try:
        return _get_runner().reattach(job_id)
    except KeyError:
        raise HTTPException(404, f"Unknown job {job_id}.") from None


@api.post("/api/create/{job_id}/stop")
def create_stop(job_id: str):
    try:
        return _get_runner().stop(job_id)
    except KeyError:
        raise HTTPException(404, f"Unknown job {job_id}.") from None


@api.get("/api/jobs")
def jobs():
    return {"jobs": _get_runner().list_jobs()}


_JOB_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _trait_job_dir(job_id: str) -> Path:
    """jobs/<id>/ for a safe id, else 404. Never leaves jobs/."""
    runner = _get_runner()
    if not _JOB_ID_RE.fullmatch(job_id or ""):
        raise HTTPException(404, f"Unknown trait {job_id}.")
    cand = (runner.jobs_dir / job_id).resolve()
    try:
        cand.relative_to(runner.jobs_dir.resolve())
    except (OSError, ValueError):
        raise HTTPException(404, f"Unknown trait {job_id}.") from None
    if not (runner.jobs_dir / job_id / "job.json").is_file():
        raise HTTPException(404, f"Unknown trait {job_id}.")
    return runner.jobs_dir / job_id


def _card_summary(card: dict | None) -> dict | None:
    """Contract summary of one report card (passed, mean, CI)."""
    if card is None:
        return None
    return {"passed": card["passed"], "trait_mean": card["trait_mean"],
            "trait_ci": list(card["trait_ci"])}


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
        trait_dir = runner.jobs_dir / job["id"] / "trait"
        try:
            rep = json.loads((trait_dir / "report.json").read_text())
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
                      if (trait_dir / "knob.lobot").is_file() else None),
            "delete_eligible": bool(frozen) and (
                trait_dir / "direction.safetensors").is_file(),
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


@api.post("/api/drafts")
def draft_create(req: DraftRequest):
    """Draft model-written parts + 20 candidate replies (C3)."""
    allowed, reason = guard_mod.draft_description_allowed(
        req.description)
    if not allowed:
        raise HTTPException(422, reason)
    if REPLAY:
        raise HTTPException(503, "Drafts need the model. "
                                 "Replay mode has no recordings for them.")
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    eng = get_engine()
    if not eng.lock.acquire(blocking=False):
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    try:
        if eng.model is None or eng.model_id != engine._active_model_id:
            eng.load(engine._active_model_id)
        try:
            if req.drafter == "muse":
                parts = _muse().draft_parts(req.description)
            else:
                parts = drafts_mod.draft_family(eng, req.description,
                                                seed=req.seed)
        except drafts_mod.DraftError as e:
            raise HTTPException(422, str(e)) from None
        try:
            replies = drafts_mod.candidate_replies(
                eng, parts, prompt_bank.PROMPT_BANK, seed=req.seed)
        except drafts_mod.DraftError as e:
            raise HTTPException(500, str(e)) from None
        draft_id = drafts_mod.new_draft_id(ROOT / "store",
                                           req.description)
        record = drafts_mod.build_draft_record(
            draft_id, req.description, parts, eng.model_id,
            eng.revision)
        drafts_mod.save_draft(ROOT / "store", draft_id, record,
                              replies)
        return {"id": draft_id, "description": record["description"],
                "parts": {k: record[k] for k in
                          ("pos", "neg", "judge", "baselines")},
                "replies": [drafts_mod.public_reply(r) for r in replies],
                "labeled": 0, "total": len(replies)}
    finally:
        eng.lock.release()


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


@api.post("/api/drafts/{draft_id}/prelabel")
def draft_prelabel(draft_id: str):
    """Muse Spark label suggestions; the user still saves the labels."""
    record, replies, labels = _draft_or_404(draft_id)
    got = _muse_call(_muse().prelabel, record["description"], replies)
    return {"id": draft_id, "suggestions": got, "total": len(replies),
            "suggested": len(got), "labeled": len(labels)}


@api.post("/api/traits/{job_id}/explain")
def trait_explain(job_id: str):
    """Plain-words explanation of a finished report (Muse Spark).

    Cached in the job dir; the numbers stay the report's own.
    """
    job_dir = _trait_job_dir(job_id)
    cached = job_dir / "trait" / "explanation.txt"
    if cached.is_file():
        return {"id": job_id, "explanation": cached.read_text().strip(),
                "cached": True}
    report = job_dir / "trait" / "report.json"
    if not report.is_file():
        raise HTTPException(404, f"No report for {job_id} yet.")
    text = _muse_call(_muse().explain, json.loads(report.read_text()))
    cached.write_text(text + "\n")
    return {"id": job_id, "explanation": text, "cached": False}


_agents: dict[str, dict] = {}
_agent_lock = threading.Lock()


@api.post("/api/agent")
def agent_start(req: AgentRequest):
    """Start the Create agent (Muse Spark + local gates) in the background."""
    allowed, reason = guard_mod.draft_description_allowed(req.description)
    if not allowed:
        raise HTTPException(422, reason)
    if REPLAY or NO_MODEL:
        raise HTTPException(503, "The agent needs the model loaded.")
    _need_torch(get_engine(), "create")
    muse = _muse()
    if not _agent_lock.acquire(blocking=False):
        raise HTTPException(409, "An agent run is in progress.")
    eng = get_engine()
    if not eng.lock.acquire(blocking=False):
        _agent_lock.release()
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    from model_lobotomy import agent as agent_mod
    cancel = threading.Event()
    state = {"status": "running", "steps": [], "result": None,
             "error": None, "cancel": cancel}

    def progress(_kind, row):
        state["steps"].append(row)

    def work():
        try:
            if eng.model is None or eng.model_id != engine._active_model_id:
                eng.load(engine._active_model_id)
            lane = None
            if req.lane == "llama":
                from model_lobotomy import llama_backend
                lane = trait_eval.LlamaLane(eng, eng.spec,
                                            llama_backend.CreateDaemon(
                                                eng.model_id, eng.revision,
                                                eng.tokenizer))
            ag = agent_mod.CreateAgent(
                eng, muse, store=_get_runner().store, lane=lane,
                max_rounds=req.max_rounds, cancel=cancel,
                progress=progress)
            state["id"] = ag.id
            state["result"] = ag.run(req.description)
            state["status"] = state["result"]["status"]
        except agent_mod.AgentStopped:
            state["status"] = "stopped"
        except Exception as e:  # noqa: BLE001 - reported to the client
            state["status"] = "failed"
            state["error"] = f"{type(e).__name__}: {e}"[:2000]
        finally:
            eng.lock.release()
            _agent_lock.release()

    run_id = uuid.uuid4().hex[:12]
    _agents[run_id] = state
    threading.Thread(target=work, daemon=True).start()
    return {"run_id": run_id, "status": "running"}


@api.get("/api/agent/{run_id}")
def agent_status(run_id: str):
    state = _agents.get(run_id)
    if state is None:
        raise HTTPException(404, f"Unknown agent run {run_id}.")
    return {k: v for k, v in state.items() if k != "cancel"}


@api.post("/api/agent/{run_id}/stop")
def agent_stop(run_id: str):
    state = _agents.get(run_id)
    if state is None:
        raise HTTPException(404, f"Unknown agent run {run_id}.")
    state["cancel"].set()
    return {"run_id": run_id, "stopping": True}


@api.get("/api/drafts/{draft_id}")
def draft_get(draft_id: str):
    """One draft: parts + labeling queue with labels so far (C3)."""
    record, replies, labels = _draft_or_404(draft_id)
    rows = []
    for entry in replies:
        row = drafts_mod.public_reply(entry)
        row["label"] = labels.get(entry["reply_id"])
        rows.append(row)
    return {"id": draft_id, "description": record["description"],
            "parts": {k: record[k] for k in
                      ("pos", "neg", "judge", "baselines")},
            "replies": rows, "labeled": len(labels),
            "total": len(replies)}


@api.post("/api/drafts/{draft_id}/labels")
def draft_label(draft_id: str, req: LabelsRequest):
    """Submit user labels for candidate replies (C3)."""
    _record, replies, labels = _draft_or_404(draft_id)
    reply_ids = {r["reply_id"] for r in replies}
    try:
        merged = drafts_mod.merge_labels(
            labels, [row.model_dump() for row in req.labels],
            reply_ids)
    except drafts_mod.DraftError as e:
        raise HTTPException(400, str(e)) from None
    drafts_mod.save_labels(ROOT / "store", draft_id, merged)
    return {"id": draft_id, "labeled": len(merged),
            "total": len(replies)}


@api.post("/api/drafts/{draft_id}/validate")
def draft_validate(draft_id: str):
    """Judge-vs-user agreement; registers the family on pass (C3)."""
    record, replies, labels = _draft_or_404(draft_id)
    if len(labels) < len(replies):
        raise HTTPException(
            422, f"Label all {len(replies)} replies first "
                 f"({len(labels)} labeled).")
    if REPLAY:
        raise HTTPException(503, "Validation needs the model. "
                                 "Replay mode has no recordings for it.")
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    eng = get_engine()
    if not eng.lock.acquire(blocking=False):
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    try:
        if eng.model is None or eng.model_id != engine._active_model_id:
            eng.load(engine._active_model_id)
        from model_lobotomy import trait_eval as trait_eval_mod
        spec = engine.active_spec()
        scores = {}
        for entry in replies:
            scores[entry["reply_id"]] = trait_eval_mod.judge_score(
                eng.model, eng.tokenizer, spec, entry["prompt"],
                entry["reply"], record["judge"])
        got = drafts_mod.label_agreement(labels, scores)
        if got["fraction"] < drafts_mod.AGREEMENT_GATE:
            disag = [rid for rid in sorted(labels)
                     if (scores[rid] >= 0.5) != bool(labels[rid])]
            raise HTTPException(422, {
                "registered": False,
                "agreement": round(got["fraction"], 4),
                "gate": drafts_mod.AGREEMENT_GATE,
                "disagreements": disag,
            })
        runner = _get_runner()
        store = runner.store
        fam = trait_store.TraitFamily(
            name=draft_id, pos=record["pos"], neg=record["neg"],
            judge=record["judge"],
            baselines=tuple(record["baselines"]))
        try:
            store.register(fam)
        except ValueError:
            raise HTTPException(422, f"A family named '{draft_id}' "
                                     f"already exists.") from None
        store.mark_validated(draft_id, n_labels=got["n"],
                             agreement=round(got["fraction"], 4),
                             note="C3 open-description validation")
        store.save()
        return {"registered": True, "family": draft_id,
                "agreement": round(got["fraction"], 4),
                "gate": drafts_mod.AGREEMENT_GATE}
    finally:
        eng.lock.release()


@api.get("/api/traits")
def traits(model_id: str | None = None):
    """Finished traits with evidence summaries (E1). No model needed."""
    items = _trait_items()
    if model_id is not None:
        items = [t for t in items if t["model_id"] == model_id]
    return {"traits": items}


@api.get("/api/traits/{job_id}/report")
def trait_report(job_id: str):
    """One trait's report.json as is (E1)."""
    path = _trait_job_dir(job_id) / "trait" / "report.json"
    if not path.is_file():
        raise HTTPException(404, f"Unknown trait {job_id}.")
    return json.loads(path.read_text())


@api.get("/api/traits/{job_id}/report.md")
def trait_report_md(job_id: str):
    """One trait's report.md as Markdown (E1)."""
    path = _trait_job_dir(job_id) / "trait" / "report.md"
    if not path.is_file():
        raise HTTPException(404, f"Unknown trait {job_id}.")
    return FileResponse(path, media_type="text/markdown")


@api.get("/api/traits/{job_id}/patch")
def trait_patch(job_id: str):
    """One trait's knob.lobot patch as a download (E1)."""
    job_dir = _trait_job_dir(job_id)
    path = job_dir / "trait" / "knob.lobot"
    if not path.is_file():
        raise HTTPException(404, f"Unknown trait {job_id}.")
    item = next((t for t in _trait_items() if t["id"] == job_id), None)
    if item is not None:
        family, version = item["family"], item["version"]
    else:
        family, version = job_id, 1
        rep_path = job_dir / "trait" / "report.json"
        if rep_path.is_file():
            try:
                family = json.loads(rep_path.read_text()).get(
                    "trait", job_id)
            except ValueError:
                pass
    return FileResponse(path, media_type="application/zip",
                        filename=f"{family}-v{version}.lobot")


@api.post("/api/update")
def update(req: UpdateRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(400, "The prompt is empty. Type a prompt first.")
    if not req.doses or len(req.doses) > MAX_DOSES:
        raise HTTPException(400, f"Give 1 to {MAX_DOSES} doses.")
    spec = engine.active_spec()
    if req.trait is not None and (req.layer is not None
                                  or req.direction is not None):
        raise HTTPException(400, "Give a trait or an explicit "
                                 "layer/direction/scale, not both.")
    if req.trait is not None:
        knob = _latest_knob(req.trait)
        if knob is None:
            raise HTTPException(404, f"No finished Create job for trait "
                                     f"'{req.trait}' with a saved direction. "
                                     f"Create the trait first.")
        layer, direction, scale = knob["layer"], knob["direction"], None
    else:
        if req.layer is None or req.direction is None or req.scale is None:
            raise HTTPException(400, "Explicit updates need layer, "
                                     "direction and scale.")
        layer, direction, scale = req.layer, req.direction, req.scale
    if not 0 <= layer < spec.layers:
        raise HTTPException(400, f"Layer {layer} out of bounds "
                                 f"[0,{spec.layers}).")
    if len(direction) != spec.hidden:
        raise HTTPException(400, f"Direction has {len(direction)} dims, "
                                 f"model needs {spec.hidden}.")
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    if get_engine().lock.locked():
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    _cancel.clear()
    return StreamingResponse(
        _update_stream(prompt, layer, direction, scale, list(req.doses),
                       min(req.max_new_tokens, steer.MAX_NEW_TOKENS_CAP),
                       spec, guard=req.guard,
                       research_mode=req.research_mode),
        media_type="application/x-ndjson")


def _delete_edits(knob: dict):
    """Ablation EditSet from a saved knob (DL)."""
    return EditSet(ablate=(Ablation(
        int(knob["layer"]), tuple(knob["direction"])),))


@api.post("/api/delete")
def delete(req: DeleteRequest):
    """Run one prompt clean vs trait-ablated, two columns (DL)."""
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(400, "The prompt is empty. Type a prompt first.")
    knob = _latest_knob(req.trait)
    if knob is None:
        raise HTTPException(422, f"Delete is not eligible for "
                                 f"'{req.trait}': no finished Create job "
                                 f"with a saved direction.")
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    _need_torch(get_engine(), "ablate")
    if get_engine().lock.locked():
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    _cancel.clear()
    return StreamingResponse(
        _delete_stream(prompt, req.trait, knob,
                       min(req.max_new_tokens, steer.MAX_NEW_TOKENS_CAP),
                       guard=req.guard,
                       research_mode=req.research_mode),
        media_type="application/x-ndjson")


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


@api.post("/api/delete/check")
def delete_check(req: DeleteCheckRequest):
    """Ablation report card with side effects + guard re-check (DL)."""
    knob = _latest_knob(req.trait)
    if knob is None:
        raise HTTPException(422, f"Delete is not eligible for "
                                 f"'{req.trait}': no finished Create job "
                                 f"with a saved direction.")
    prompts = None
    if req.prompts is not None:
        prompts = [p for p in req.prompts if p and p.strip()]
        if not prompts:
            raise HTTPException(400, "Give at least one prompt.")
        if len(prompts) > 24:
            raise HTTPException(400, "Give at most 24 prompts.")
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    _need_torch(get_engine(), "ablate")
    eng = get_engine()
    if not eng.lock.acquire(blocking=False):
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    try:
        t_all = time.perf_counter()
        if eng.model is None or eng.model_id != engine._active_model_id:
            eng.load(engine._active_model_id)
        spec = engine.active_spec()
        medians, _ = _calibrate(eng, [int(knob["layer"])])
        try:
            cfg = trait_eval.family_cfg(req.trait)
        except KeyError:
            raise HTTPException(422, f"Delete is not eligible for "
                                     f"'{req.trait}': no registered family.") from None
        try:
            out = trait_eval.delete_check(
                eng, spec, cfg, knob, medians, prompt_bank.PROMPT_BANK,
                prompts=prompts, cancel=_cancel)
        except ValueError as e:
            raise HTTPException(400, f"{e}.") from None
        out["seconds"] = round(time.perf_counter() - t_all, 1)
        return out
    finally:
        eng.lock.release()


_install_lock = threading.Lock()
_INSTALL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@api.get("/api/install/targets")
def install_targets():
    """Probe llama.cpp and Ollama. Needs no weights (IN)."""
    return {"targets": [install_mod.probe(t)
                        for t in install_mod.TARGETS]}


@api.post("/api/install")
def install_patch(req: InstallRequest):
    """Export a trait patch and install it into one target (IN)."""
    if req.target not in install_mod.TARGETS:
        raise HTTPException(400, f"Unknown target {req.target!r}; "
                                 f"use one of {install_mod.TARGETS}.")
    if not req.model_ref.strip():
        raise HTTPException(400, "Give a model_ref.")
    job_dir = _trait_job_dir(req.trait_id)
    patch_path = job_dir / "trait" / "knob.lobot"
    if not patch_path.is_file():
        raise HTTPException(404, f"Unknown trait {req.trait_id}.")
    if not _install_lock.acquire(blocking=False):
        raise HTTPException(409, "An install or verify is running. "
                                 "Wait for it to finish.")
    try:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        install_id = (f"{req.trait_id}-{req.target}-{stamp}")
        out_dir = install_mod.INSTALLS_ROOT / install_id
        got = install_mod.install(req.target, patch_path, out_dir,
                                  req.model_ref.strip())
        if not got["ok"]:
            raise HTTPException(422, got["detail"])
        record = {"install_id": install_id,
                  "trait_id": req.trait_id,
                  "target": req.target,
                  "model_ref": req.model_ref.strip(),
                  "run": {k: got[k] for k in got
                          if k not in ("ok", "detail")},
                  "detail": got["detail"]}
        install_mod.save_install(record)
        return record
    finally:
        _install_lock.release()


@api.post("/api/install/{install_id}/verify")
def install_verify(install_id: str, req: InstallVerifyRequest):
    """Run with and without the installed patch; compare (IN)."""
    if not _INSTALL_ID_RE.fullmatch(install_id or ""):
        raise HTTPException(404, f"Unknown install {install_id}.")
    prompt = req.prompt.strip()
    if req.judge not in (None, "local"):
        raise HTTPException(400, "judge must be null or 'local'.")
    if req.judged_n:
        if req.judge != "local":
            raise HTTPException(400, "judged_n needs judge 'local'.")
        if not 1 <= req.judged_n <= INSTALL_JUDGED_MAX:
            raise HTTPException(400, f"judged_n must be 1 to "
                                     f"{INSTALL_JUDGED_MAX}.")
    elif not prompt:
        raise HTTPException(400, "The prompt is empty. "
                                 "Type a prompt first.")
    record = install_mod.load_install(install_id)
    if record is None:
        raise HTTPException(404, f"Unknown install {install_id}.")
    if req.judged_n:
        return _install_verify_judged(install_id, record, req)
    if not _install_lock.acquire(blocking=False):
        raise HTTPException(409, "An install or verify is running. "
                                 "Wait for it to finish.")
    try:
        t_all = time.perf_counter()
        max_new = min(max(int(req.max_new_tokens), 1), 256)
        base = install_mod.generate(record["target"], record["run"],
                                    prompt, patched=False,
                                    max_new=max_new)
        if not base["ok"]:
            raise HTTPException(422, f"Unpatched run failed: "
                                     f"{base['detail']}")
        patched = install_mod.generate(record["target"], record["run"],
                                       prompt, patched=True,
                                       max_new=max_new)
        if not patched["ok"]:
            raise HTTPException(422, f"Patched run failed: "
                                     f"{patched['detail']}")
        out = {"install_id": install_id, "prompt": prompt,
               "base_text": base["text"],
               "patched_text": patched["text"],
               "differs": patched["text"] != base["text"],
               "comparison": None,
               "seconds": round(time.perf_counter() - t_all, 1)}
        if req.judge == "local":
            out["comparison"] = _install_compare(record, prompt,
                                                 base["text"],
                                                 patched["text"])
            out["seconds"] = round(time.perf_counter() - t_all, 1)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (install_mod.INSTALLS_ROOT / install_id /
         f"verify-{stamp}.json").write_text(json.dumps(out, indent=2))
        return out
    finally:
        _install_lock.release()


def _final_prompts(record: dict, n: int) -> list[str]:
    """First n final-split prompts of the install's Create job."""
    report = _trait_job_dir(record["trait_id"]) / "trait" / "report.json"
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
            job_dir / "trait" / "knob.lobot")["meta"]
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


@api.post("/api/verify")
def verify(req: VerifyRequest):
    """Re-run a trait patch's shipped evidence (RV)."""
    job_dir = _trait_job_dir(req.trait_id)
    patch_path = job_dir / "trait" / "knob.lobot"
    if not patch_path.is_file():
        raise HTTPException(422, f"Trait {req.trait_id} has no saved "
                                 f"patch to verify.")
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded "
                                 "(MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to verify.")
    if get_engine().lock.locked():
        raise HTTPException(409, "A run is in progress. "
                                 "Wait for it to finish.")
    _need_torch(get_engine(), "verify")
    _verify_pin(patch_path)
    _cancel.clear()
    return StreamingResponse(_verify_stream(patch_path, req.quick),
                             media_type="application/x-ndjson")


@api.post("/api/verify/upload")
async def verify_upload(file: UploadFile = File(...)):
    """Verify an uploaded .lobot file (RV). Never unpickles."""
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded "
                                 "(MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to verify.")
    if get_engine().lock.locked():
        raise HTTPException(409, "A run is in progress. "
                                 "Wait for it to finish.")
    _need_torch(get_engine(), "verify")
    data = b""
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        data += chunk
        if len(data) > UPLOAD_BYTES_CAP:
            raise HTTPException(413, "Patch file is over 64 MB.")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.lobot"
        probe.write_bytes(data)
        _verify_pin(probe)
    _cancel.clear()
    return StreamingResponse(_verify_stream(data, True),
                             media_type="application/x-ndjson")


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
        path = job_dir / "trait" / "knob.lobot"
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
        guard = guard_mod.Guard()
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


@api.post("/api/combine/check")
def combine_check(req: CombineCheckRequest):
    """Pair matrix + install order for saved patches (CB)."""
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded "
                                 "(MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to combine.")
    names, patches = _combine_patches(req.patch_ids)
    if any(p["meta"].get("edit_kind") == "ablate" for p in patches):
        _need_torch(get_engine(), "ablate")
    if get_engine().lock.locked():
        raise HTTPException(409, "A run is in progress. "
                                 "Wait for it to finish.")
    return _combine_matrix(names, patches, req.max_new_tokens,
                           req.seed)


@api.get("/api/combine/installed")
def combine_installed():
    """Pair matrix for the traits installed via /api/install (CB)."""
    root = install_mod.INSTALLS_ROOT
    found: dict[str, float] = {}
    if root.is_dir():
        for rec in sorted(root.glob("*/install.json"),
                          key=lambda p: p.stat().st_mtime):
            try:
                tid = json.loads(rec.read_text()).get("trait_id")
            except ValueError:
                continue
            if tid and tid not in found:
                found[tid] = rec.stat().st_mtime
    if not found:
        return {"installed": [], "matrix": None}
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded "
                                 "(MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to combine.")
    names, patches = _combine_patches(list(found))
    if any(p["meta"].get("edit_kind") == "ablate" for p in patches):
        _need_torch(get_engine(), "ablate")
    if get_engine().lock.locked():
        raise HTTPException(409, "A run is in progress. "
                                 "Wait for it to finish.")
    return {"installed": list(found),
            "matrix": _combine_matrix(names, patches, 24, 0)}


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


@api.post("/api/neurons")
def neurons(req: NeuronsRequest):
    """Click-a-word (R1, R2): top neurons at one prompt or reply token."""
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(400, "The prompt is empty. Type a prompt first.")
    if len(req.rows) > engine.MAX_ROWS:
        raise HTTPException(400, f"Too many rows. The limit is {engine.MAX_ROWS}.")
    rows = [RecipeRow(layer=r.layer, neuron=r.neuron, dose=r.dose) for r in req.rows]
    spec = engine.active_spec()
    try:
        if rows:
            steer.validate_recipe(rows, spec)
    except ValueError as e:
        raise HTTPException(400, f"Invalid row: {e}.") from None
    knob = None
    dose = 1.0
    if req.trait is not None:
        knob = _latest_knob(req.trait)
        if knob is None:
            raise HTTPException(404, f"No finished Create job for trait "
                                     f"'{req.trait}' with a saved direction. "
                                     f"Create the trait first.")
        dose = req.dose if req.dose is not None else float(knob["dose"])
    if (req.token_index is None) == (req.reply_char is None):
        raise HTTPException(400, "Give exactly one of token_index and reply_char.")
    if req.reply_char is not None and not (req.reply and 0 <= req.reply_char < len(req.reply)):
        raise HTTPException(400, "reply_char needs a reply and must fall inside it.")
    top_k = min(max(int(req.top_k), 1), 32)
    window = min(max(int(req.window), 0), 16)
    if REPLAY:
        raise HTTPException(503, "Replay mode loads no weights, and the neurons at a word "
                                 "need the model. Restart without MODEL_LOBOTOMY_REPLAY.")
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    _need_torch(get_engine(), "capture")
    eng = get_engine()
    if not eng.lock.acquire(blocking=False):
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    try:
        if eng.model is None or eng.model_id != engine._active_model_id:
            eng.load(engine._active_model_id)
        preset = engine.active_preset()
        steer.check_preset_binding(preset, engine._active_model_id)
        layers = sorted({r.layer for r in rows}
                        | ({int(knob["layer"])} if knob else set()))
        medians, _ = _calibrate(eng, layers)
        vec = ()
        if knob is not None:
            vec = (_trait_vector(
                knob, dose, float(medians[int(knob["layer"])])),)
        edits = EditSet(neuron=EditSet.from_rows(rows).neuron,
                        vector=vec)
        gv = _verified_guard(
            eng, edits, medians,
            guard_mod.should_enforce(req.guard, req.research_mode))
        if gv.blocked:
            return {"blocked": True, "guard": _guard_event_dict(gv)}
        wrapped = steer.wrap_prompt_notk(eng.tokenizer, prompt, spec)
        ids = _token_ids(eng.tokenizer, wrapped)
        reply_start = None
        if req.reply:
            full = _token_ids(eng.tokenizer, wrapped + req.reply)
            reply_start = len(ids)
            for i, (a, b) in enumerate(zip(full, ids)):
                if a != b:
                    reply_start = i
                    break
            ids = full
        token_index = req.token_index
        if req.reply_char is not None:
            token_index = _reply_token_at(eng.tokenizer, ids, reply_start, req.reply_char)
        if not 1 <= token_index < len(ids):
            raise HTTPException(400, f"token_index {token_index} out of "
                                     f"bounds [1,{len(ids)}).")
        if req.reply:
            mats = _capture_reply_mats(eng, edits, medians, ids, spec)
        else:
            # capture() wraps once itself and drops position 0 (BOS):
            # matrix row r is wrapped token r + 1.
            activ = eng.capture_with_edits(
                [prompt], list(range(spec.layers)), edits, medians,
                point="mlp_in", reduce="none")
            mats = {L: activ[L][0] for L in range(spec.layers)}
        neurons = _rank_neurons(
            eng, mats, ids, token_index, top_k, window)
        return {"blocked": False, "model_id": engine._active_model_id,
                "prompt": prompt,
                "tokens": [eng.tokenizer.decode([i],
                                                skip_special_tokens=True)
                           for i in ids],
                "token_index": token_index, "reply_start": reply_start,
                "neurons": neurons}
    finally:
        eng.lock.release()


@api.get("/api/neuron/{layer}/{neuron}")
def neuron_card(layer: int, neuron: int, k: int = Query(12, ge=1, le=50)):
    """One neuron card: pushes_toward plus cached fires_on (R2)."""
    spec = engine.active_spec()
    if not 0 <= layer < spec.layers or not 0 <= neuron < spec.intermediate:
        raise HTTPException(400, f"Neuron L{layer}N{neuron} is out of "
                                 f"bounds [0,{spec.layers})x"
                                 f"[0,{spec.intermediate}).")
    cache_dir = ROOT / "store" / "reader" / _model_slug(
        engine._active_model_id)
    if REPLAY:
        cards_path = cache_dir / "cards.json"
        if not cards_path.exists():
            raise HTTPException(
                503, "Neuron cards need the model, or a recorded "
                     "cards.json for replay mode.")
        cards = json.loads(cards_path.read_text())
        card = cards.get(f"{layer}:{neuron}")
        if card is None:
            raise HTTPException(
                404, f"No recorded card for neuron L{layer}N{neuron}.")
        return card
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    _need_torch(get_engine(), "reader")
    eng = get_engine()
    if not eng.lock.acquire(blocking=False):
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    try:
        if eng.model is None or eng.model_id != engine._active_model_id:
            eng.load(engine._active_model_id)
        pushes = reader.pushes_toward(eng, layer, neuron, k=k)
        fires, built, tokens = None, False, 0
        if ((cache_dir / "fires_on.npz").exists()
                and (cache_dir / "sample.json").exists()):
            sample = json.loads((cache_dir / "sample.json").read_text())
            tokens = len(sample.get("tokens", []))
            fires = reader.fires_on(cache_dir, layer, neuron, k=k)
            built = True
        return {"model_id": engine._active_model_id, "layer": layer,
                "neuron": neuron, "pushes_toward": pushes,
                "fires_on": fires,
                "fires_on_cache": {"built": built, "tokens": tokens}}
    finally:
        eng.lock.release()


@api.get("/api/find")
def find(word: str = "", layers: str | None = None,
         k: int = Query(20, ge=1, le=50)):
    """Neurons that push one word hardest (R2)."""
    spec = engine.active_spec()
    layer_list = _parse_layers(layers, spec)
    if not word or not word.strip():
        raise HTTPException(400, "The word is empty. Type a word first.")
    if REPLAY:
        raise HTTPException(503, "Find needs the model. Replay mode has no recordings for it.")
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    _need_torch(get_engine(), "reader")
    eng = get_engine()
    if not eng.lock.acquire(blocking=False):
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    try:
        if eng.model is None or eng.model_id != engine._active_model_id:
            eng.load(engine._active_model_id)
        try:
            return reader.find_neurons(eng, word, layer_list, k=k)
        except ValueError as e:
            raise HTTPException(400, f"{e}.") from None
    finally:
        eng.lock.release()


@api.post("/api/lift")
def lift(req: LiftRequest):
    """Steering lift for one neuron (R2)."""
    spec = engine.active_spec()
    if not 0 <= req.layer < spec.layers or not 0 <= req.neuron < spec.intermediate:
        raise HTTPException(400, f"Neuron L{req.layer}N{req.neuron} is out of "
                                 f"bounds [0,{spec.layers})x"
                                 f"[0,{spec.intermediate}).")
    prompts = req.prompts
    if prompts is None:
        path = ROOT / "store" / "reader" / "lift_prompts.json"
        if not path.exists():
            raise HTTPException(500, "The lift prompts are missing from the server.")
        prompts = json.loads(path.read_text())
    if REPLAY:
        raise HTTPException(503, "Lift needs the model. Replay mode has no recordings for it.")
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    _need_torch(get_engine(), "reader")
    eng = get_engine()
    if not eng.lock.acquire(blocking=False):
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    try:
        if eng.model is None or eng.model_id != engine._active_model_id:
            eng.load(engine._active_model_id)
        medians, _ = _calibrate(eng, [req.layer])
        try:
            return reader.steer_lift(eng, req.layer, req.neuron, req.dose,
                                     req.words, prompts, medians)
        except ValueError as e:
            raise HTTPException(400, f"{e}.") from None
    finally:
        eng.lock.release()


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


@api.get("/api/features")
def features():
    """Trained SAEs for the active model (F1). No model needed."""
    saes = []
    for layer, _dir, cfg in _sae_layers():
        saes.append({
            "model_id": cfg.get("model_id", engine._active_model_id),
            "layer": layer,
            "features": int(cfg.get("features", 0)),
            "top_k": int(cfg.get("top_k", 0)),
            "live": cfg.get("live"),
            "dead": cfg.get("dead"),
            "mse": cfg.get("mse_last"),
        })
    return {"saes": saes}


@api.get("/api/feature/{layer}/{index}")
def feature_card(layer: int, index: int, k: int = Query(12, ge=1, le=50)):
    """One SAE feature card: pushes plus cached fires (F1)."""
    spec = engine.active_spec()
    if not 0 <= layer < spec.layers:
        raise HTTPException(400, f"Layer {layer} is out of bounds "
                                 f"[0,{spec.layers}).")
    found = next((d for L, d, _c in _sae_layers() if L == layer), None)
    if found is None:
        raise HTTPException(404, f"No trained SAE for layer {layer} on "
                                 f"{engine._active_model_id}. Train one with "
                                 f"scripts/train_sae.py.")
    if REPLAY:
        raise HTTPException(503, "Feature cards need the model. "
                                 "Replay mode has no recordings for them.")
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    _need_torch(get_engine(), "sae")
    eng = get_engine()
    if not eng.lock.acquire(blocking=False):
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    try:
        if eng.model is None or eng.model_id != engine._active_model_id:
            eng.load(engine._active_model_id)
        sae, _cfg = _load_sae_cached(found)
        if not 0 <= index < sae.features:
            raise HTTPException(400, f"Feature {index} is out of bounds "
                                     f"[0,{sae.features}).")
        return sae_mod.feature_card(eng, sae, layer, index,
                                    cache_dir=found, k=k)
    finally:
        eng.lock.release()


@api.post("/api/feature-lift")
def feature_lift(req: FeatureLiftRequest):
    """Steering lift for one SAE feature (F1)."""
    spec = engine.active_spec()
    if not 0 <= req.layer < spec.layers:
        raise HTTPException(400, f"Layer {req.layer} is out of bounds "
                                 f"[0,{spec.layers}).")
    found = next((d for L, d, _c in _sae_layers() if L == req.layer), None)
    if found is None:
        raise HTTPException(404, f"No trained SAE for layer {req.layer} on "
                                 f"{engine._active_model_id}. Train one with "
                                 f"scripts/train_sae.py.")
    prompts = req.prompts
    if prompts is None:
        path = ROOT / "store" / "reader" / "lift_prompts.json"
        if not path.exists():
            raise HTTPException(500, "The lift prompts are missing from the server.")
        prompts = json.loads(path.read_text())
    if REPLAY:
        raise HTTPException(503, "Feature lift needs the model. "
                                 "Replay mode has no recordings for it.")
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    _need_torch(get_engine(), "sae")
    eng = get_engine()
    if not eng.lock.acquire(blocking=False):
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    try:
        if eng.model is None or eng.model_id != engine._active_model_id:
            eng.load(engine._active_model_id)
        sae, _cfg = _load_sae_cached(found)
        if not 0 <= req.index < sae.features:
            raise HTTPException(400, f"Feature {req.index} is out of bounds "
                                     f"[0,{sae.features}).")
        words = req.words
        if words is None:
            words = sae_mod.top_words(reader.pushes_of_direction(
                eng, sae_mod.feature_direction(sae, req.index),
                layer=req.layer))
        if not words:
            raise HTTPException(422, "Give at least one word.")
        medians, _ = _calibrate(eng, [req.layer])
        try:
            return reader.steer_lift_vector(
                eng, req.layer, sae_mod.feature_direction(sae, req.index),
                req.dose, words, prompts, medians)
        except ValueError as e:
            raise HTTPException(400, f"{e}.") from None
    finally:
        eng.lock.release()


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


@api.post("/api/chat/start")
def chat_start(req: ChatStartRequest):
    """Open a chat session; its edits persist across turns (E2)."""
    if len(req.rows) > engine.MAX_ROWS:
        raise HTTPException(400, f"Too many rows. The limit is {engine.MAX_ROWS}.")
    rows = [RecipeRow(layer=r.layer, neuron=r.neuron, dose=r.dose) for r in req.rows]
    spec = engine.active_spec()
    try:
        if rows:
            steer.validate_recipe(rows, spec)
    except ValueError as e:
        raise HTTPException(400, f"Invalid row: {e}.") from None
    dose = 1.0
    if req.trait is not None:
        knob = _latest_knob(req.trait)
        if knob is None:
            raise HTTPException(404, f"No finished Create job for trait "
                                     f"'{req.trait}' with a saved direction. "
                                     f"Create the trait first.")
        dose = req.dose if req.dose is not None else float(knob["dose"])
    dial_traits = list(req.dial_traits or [])
    if len(dial_traits) > MAX_CHAT_DIALS:
        raise HTTPException(400, f"Too many dial traits. "
                                 f"The limit is {MAX_CHAT_DIALS}.")
    for trait in dial_traits:
        if _latest_knob(trait) is None:
            raise HTTPException(404, f"No finished Create job for trait "
                                     f"'{trait}' with a saved direction. "
                                     f"Create the trait first.")
    if dial_traits:
        _need_torch(get_engine(), "dial")
    if len(_chats) >= MAX_CHAT_SESSIONS:
        oldest = min(_chats, key=lambda k: _chats[k]["created"])
        del _chats[oldest]
    sid = uuid.uuid4().hex[:16]
    _chats[sid] = {"messages": [], "rows": rows, "trait": req.trait,
                   "dose": dose, "dial_traits": dial_traits,
                   "turn_records": [],
                   "max_new_tokens": min(req.max_new_tokens,
                                         steer.MAX_NEW_TOKENS_CAP),
                   "guard": req.guard, "research_mode": req.research_mode,
                   "model_id": engine._active_model_id,
                   "created": time.time()}
    return {"session_id": sid, "model_id": engine._active_model_id}


@api.get("/api/chat/{session_id}")
def chat_history(session_id: str):
    """Chat history for reattach (turns stay server-side)."""
    sess = _chats.get(session_id)
    if sess is None:
        raise HTTPException(404, f"Unknown chat {session_id}.")
    return {"session_id": session_id, "model_id": sess["model_id"],
            "turns": list(sess["messages"]),
            "dial_traits": list(sess.get("dial_traits") or []),
            "dial_turns": [
                {"reply": rec["reply"], "dials": rec["dials"]}
                for rec in sess.get("turn_records") or []]}


@api.get("/api/chat/{session_id}/transcript")
def chat_transcript(session_id: str):
    """Full replay spec for a chat: per-turn messages, rows, dials (CH)."""
    sess = _chats.get(session_id)
    if sess is None:
        raise HTTPException(404, f"Unknown chat {session_id}.")
    return {
        "model_id": sess["model_id"],
        "trait": sess.get("trait"),
        "dose": sess.get("dose", 1.0),
        "dial_traits": list(sess.get("dial_traits") or []),
        "guard": bool(sess.get("guard", False)),
        "research_mode": bool(sess.get("research_mode", False)),
        "turns": [
            {"message": rec["message"], "rows": rec["rows"],
             "max_new_tokens": rec.get("max_new_tokens"),
             "reply": rec["reply"], "dials": rec["dials"]}
            for rec in sess.get("turn_records") or []],
    }


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


@api.post("/api/chat/replay")
def chat_replay(req: ReplayRequest):
    """Re-run a transcript in a fresh session; report per-turn match (CH)."""
    data = req.transcript or {}
    turns = data.get("turns")
    if not isinstance(turns, list) or not turns:
        raise HTTPException(400, "The transcript has no turns.")
    if data.get("model_id") != engine._active_model_id:
        raise HTTPException(
            409, f"Transcript is pinned to {data.get('model_id')}; "
                 f"the active model is {engine._active_model_id}. "
                 f"Switch back or start a new chat.")
    try:
        parsed = [ReplayTurn(**t) for t in turns]
    except Exception as e:
        raise HTTPException(400, f"Bad transcript turn: {e}.") from None
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    if get_engine().lock.locked():
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    spec = engine.active_spec()
    for turn in parsed:
        rows = [RecipeRow(layer=r.layer, neuron=r.neuron, dose=r.dose)
                for r in turn.rows]
        try:
            if rows:
                steer.validate_recipe(rows, spec)
        except ValueError as e:
            raise HTTPException(400, f"Invalid row: {e}.") from None
        if not turn.message.strip():
            raise HTTPException(400, "A transcript message is empty.")
    start = ChatStartRequest(
        trait=data.get("trait"), dose=data.get("dose", 1.0),
        guard=bool(data.get("guard", False)),
        research_mode=bool(data.get("research_mode", False)),
        dial_traits=list(data.get("dial_traits") or []))
    try:
        if start.trait is not None and _latest_knob(start.trait) is None:
            raise KeyError(start.trait)
        for trait in start.dial_traits:
            if _latest_knob(trait) is None:
                raise KeyError(trait)
    except KeyError as e:
        raise HTTPException(404, f"No finished Create job for trait "
                                 f"'{e.args[0]}' with a saved direction. "
                                 f"Create the trait first.") from None
    if len(start.dial_traits) > MAX_CHAT_DIALS:
        raise HTTPException(400, f"Too many dial traits. "
                                 f"The limit is {MAX_CHAT_DIALS}.")
    if start.dial_traits:
        _need_torch(get_engine(), "dial")
    if len(_chats) >= MAX_CHAT_SESSIONS:
        oldest = min(_chats, key=lambda k: _chats[k]["created"])
        del _chats[oldest]
    sid = uuid.uuid4().hex[:16]
    _chats[sid] = {"messages": [], "rows": [], "trait": start.trait,
                   "dose": start.dose if start.trait is not None else 1.0,
                   "dial_traits": start.dial_traits, "turn_records": [],
                   "max_new_tokens": steer.MAX_NEW_TOKENS_CAP,
                   "guard": start.guard,
                   "research_mode": start.research_mode,
                   "model_id": engine._active_model_id,
                   "created": time.time()}
    results = []
    for i, (turn, want) in enumerate(zip(parsed, turns), start=1):
        rows = [RecipeRow(layer=r.layer, neuron=r.neuron, dose=r.dose)
                for r in turn.rows]
        _cancel.clear()
        reply, dials, err = None, None, None
        max_new = want.get("max_new_tokens") or steer.MAX_NEW_TOKENS_CAP
        max_new = min(int(max_new), steer.MAX_NEW_TOKENS_CAP)
        for chunk in _chat_stream(sid, turn.message, rows, max_new):
            ev = json.loads(chunk.decode())
            if ev.get("stage") == "done":
                reply = ev["result"]["reply"]
                dials = ev["result"].get("dials", {})
            elif ev.get("stage") in ("error", "blocked", "stopped"):
                err = ev.get("message", ev.get("stage"))
        reply_match = err is None and reply == want.get("reply")
        diff = _dials_max_diff(dials or {}, want.get("dials") or {})
        dials_match = err is None and diff == 0.0
        results.append({"turn": i, "match": bool(reply_match and dials_match),
                        "reply_match": bool(reply_match),
                        "dials_match": bool(dials_match),
                        "dials_max_diff": diff,
                        "error": err})
    match = all(r["match"] for r in results)
    return {"match": match, "session_id": sid, "turns": results}


@api.post("/api/chat")
def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(400, "The message is empty. Type one first.")
    sess = _chats.get(req.session_id)
    if sess is None:
        raise HTTPException(404, f"Unknown chat {req.session_id}.")
    if sess["model_id"] != engine._active_model_id:
        raise HTTPException(409, f"Chat {req.session_id} was started on "
                                 f"{sess['model_id']}; the active model is "
                                 f"{engine._active_model_id}. Switch back or "
                                 f"start a new chat.")
    rows = None
    if req.rows is not None:
        if len(req.rows) > engine.MAX_ROWS:
            raise HTTPException(400, f"Too many rows. The limit is {engine.MAX_ROWS}.")
        rows = [RecipeRow(layer=r.layer, neuron=r.neuron, dose=r.dose)
                for r in req.rows]
        try:
            if rows:
                steer.validate_recipe(rows, engine.active_spec())
        except ValueError as e:
            raise HTTPException(400, f"Invalid row: {e}.") from None
    if NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    if get_engine().lock.locked():
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    if sess.get("dial_traits"):
        _need_torch(get_engine(), "dial")
    _cancel.clear()
    max_new = sess["max_new_tokens"]
    if req.max_new_tokens is not None:
        max_new = min(req.max_new_tokens, steer.MAX_NEW_TOKENS_CAP)
    return StreamingResponse(
        _chat_stream(req.session_id, message, rows, max_new),
        media_type="application/x-ndjson")


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
        path = (runner.jobs_dir / job["id"] / "trait"
                / "direction.safetensors")
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
    dpath = runner.jobs_dir / knob["job_id"] / "trait" / "dial.json"
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
    return guard_mod.Guard().check_verified(
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
    uvicorn.run(api, host="127.0.0.1", port=int(os.environ.get("PORT", 7860)))
