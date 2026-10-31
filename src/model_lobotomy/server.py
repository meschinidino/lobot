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
    POST /api/chat/start  open a chat session (edits persist, E2)
    POST /api/chat         one chat turn, streamed tokens
    GET  /api/chat/{id}    chat history (reattach)

Full request/response shapes: docs/api.md.

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
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictBool
from starlette.responses import FileResponse, JSONResponse

from model_lobotomy import app as engine
from model_lobotomy import create_job
from model_lobotomy import drafts as drafts_mod
from model_lobotomy import guard as guard_mod
from model_lobotomy import prompt_bank
from model_lobotomy import sae as sae_mod
from model_lobotomy import steer
from model_lobotomy import trait_eval
from model_lobotomy import trait_store
from model_lobotomy.engine import EMPTY_EDITS, EditSet, Read, VectorEdit, apply_edits, get_engine
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


class ChatStartRequest(BaseModel):
    rows: list[Row] = []
    trait: str | None = None
    dose: float | None = None
    max_new_tokens: int = 150
    guard: bool = False
    research_mode: bool = False


class ChatRequest(BaseModel):
    session_id: str
    message: str
    rows: list[Row] | None = None
    max_new_tokens: int | None = None


def _state() -> dict:
    spec = engine.active_spec()
    preset = engine.active_preset()
    return {
        "active_model": engine._active_model_id,
        "warm": engine._model is not None and engine._model_id == engine._active_model_id,
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
        return steer.calibrate_layer_medians(
            eng.model, eng.tokenizer, list(preset["calibration_prompts"]),
            layers or [0],
            cache_path=ROOT / steer.calibration_path_for(
                engine._active_model_id),
            key=key, spec=spec)


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
    if len(_chats) >= MAX_CHAT_SESSIONS:
        oldest = min(_chats, key=lambda k: _chats[k]["created"])
        del _chats[oldest]
    sid = uuid.uuid4().hex[:16]
    _chats[sid] = {"messages": [], "rows": rows, "trait": req.trait,
                   "dose": dose,
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
            "turns": list(sess["messages"])}


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
        medians, _ = _calibrate(eng, layers)
        scale = None
        if sess["trait"] is not None:
            knob = _latest_knob(sess["trait"])
            scale = float(medians[int(knob["layer"])])
        edits, _ = _chat_edits(sess, scale=scale)
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
        for ev in eng.run_stream(
                message, edits, medians=medians,
                max_new_tokens=max_new, seed=engine.SEED,
                cancel=_cancel, wrapped=wrapped):
            if ev["type"] == "token":
                yield _event(stage="token", text=ev["text"])
            else:
                reply = ev["result"]
        if reply.get("cancelled"):
            yield _event(stage="stopped", after="chat")
            return
        sess["messages"] = (messages + [{"role": "assistant",
                                         "content": reply["text"]}])
        while len(sess["messages"]) // 2 > MAX_CHAT_TURNS:
            del sess["messages"][:2]
        yield _event(stage="done", result={
            "session_id": session_id, "turn": len(sess["messages"]) // 2,
            "reply": reply["text"], "tokens": reply["new_tokens"],
            "seconds": round(time.perf_counter() - t_all, 1)})
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

# Chat sessions: id -> {messages, rows, trait, dose, max_new_tokens,
# guard, research_mode, model_id, created}. In memory; a restart clears.
_chats: dict = {}
MAX_CHAT_SESSIONS = 16
MAX_CHAT_TURNS = 40


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
    cfg = trait_eval.spike.BEHAVIORS.get(trait)
    if cfg is None:
        raise KeyError(trait)
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
