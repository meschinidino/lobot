"""lobot web server: FastAPI over the engine, serves the web/ frontend.

Same before/after loop as app.py (Gradio), with a custom frontend that
follows DESIGN.md. Engine state (warm model, presets, active model) stays
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
    POST /api/update  vector Update filmstrip, per-column token NDJSON

Full request/response shapes: docs/api.md.

Run:
    uv run python server.py                            # loads weights, serves localhost:7860
    MODEL_LOBOTOMY_NO_MODEL=1 uv run python server.py  # UI only; runs return 503
"""

from __future__ import annotations

import json
import os
import threading
import time
from os.path import commonprefix
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

import app as engine
import create_job
import guard as guard_mod
import steer
import trait_store
from engine import EMPTY_EDITS, EditSet, VectorEdit, get_engine
from steer import RecipeRow

ROOT = Path(__file__).resolve().parent
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


class Row(BaseModel):
    layer: int
    neuron: int
    dose: float = Field(ge=engine.DOSE_MIN, le=engine.DOSE_MAX)


class RunRequest(BaseModel):
    prompt: str
    rows: list[Row]
    guard: bool = False
    research_mode: bool = False


class ModelRequest(BaseModel):
    model_id: str


class CreateRequest(BaseModel):
    description: str
    seed: int = 0
    guard: bool = False
    research_mode: bool = False


class UpdateRequest(BaseModel):
    prompt: str
    trait: str | None = None
    layer: int | None = None
    direction: list[float] | None = None
    scale: float | None = None
    doses: list[float]
    max_new_tokens: int = 60
    guard: bool = False
    research_mode: bool = False


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
    }


def _recordings() -> list[dict]:
    """Every recorded run as a /api/run result: example.json, then each web/sweep*.json."""
    out = []
    ex_path = WEB / "example.json"
    if ex_path.exists():
        ex = json.loads(ex_path.read_text())
        out.append({k: ex[k] for k in ("model_id", "prompt", "rows", "baseline", "steered",
                                       "split", "seconds", "recipe", "post")})
    for sw_path in sorted(WEB.glob("sweep*.json")):
        sw = json.loads(sw_path.read_text())
        for r in sw["runs"]:
            out.append({"model_id": sw["model_id"], "prompt": sw["prompt"], "rows": r["rows"],
                        "baseline": sw["baseline"],
                        "steered": {"text": r["text"], "tokens": r["tokens"]},
                        "split": r["split"], "seconds": r["seconds"],
                        "recipe": r.get("recipe", ""), "post": r.get("post", "")})
    return out


def _same_rows(a: list[dict], b: list[dict]) -> bool:
    key = lambda rows: sorted((r["layer"], r["neuron"], round(float(r["dose"]), 4)) for r in rows)
    return key(a) == key(b)


def _find_recording(model_id: str, prompt: str, rows: list[dict]) -> dict | None:
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
        yield _event(stage="complete", result={**rec, "cache_hit": True, "replay": True})
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
                    research_mode=req.research_mode),
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
                              research_mode=req.research_mode)
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


def _latest_knob(trait: str) -> dict | None:
    """Newest finished Create job for a trait with a saved direction."""
    runner = _get_runner()
    cands = [j for j in runner.list_jobs()
             if j.get("trait") == trait and j.get("status") == "done"
             and j.get("frozen")]
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


def _guard_event_dict(gv) -> dict:
    return {"blocked": gv.blocked, "would_block": gv.would_block,
            "enforced": not gv.research_mode,
            "verified": gv.full_generation,
            "min_ratio": round(gv.min_ratio, 3),
            "worst_prompt": gv.worst_prompt,
            "blocked_prompts": list(gv.blocked_prompts)}


def _verified_guard(eng, edits, medians, enforce):
    return guard_mod.Guard().check_verified(
        eng, edits, medians=medians, max_new_tokens=24, seed=engine.SEED,
        judge_fn=guard_mod.refusal_judge_fn(eng),
        research_mode=not enforce, cancel=_cancel)


def _run_stream(prompt: str, rows: list[RecipeRow], spec,
                guard: bool = False, research_mode: bool = False):
    """Stage events, then one result event. Holds the Engine lock throughout.

    The lock is taken here, not in run(): a client that disconnects before
    the first chunk never starts this generator, so nothing leaks.
    The Engine comes from get_engine() so tests inject a fake (same asserts).
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
        layers = sorted({r.layer for r in rows})
        key = steer.calibration_cache_key(
            engine._active_model_id, eng.revision, tok, str(steer.DTYPE), steer.DEVICE)
        with eng.clean():
            medians, cal_info = steer.calibrate_layer_medians(
                model, tok, calib_prompts, layers,
                cache_path=ROOT / steer.calibration_path_for(engine._active_model_id),
                key=key, spec=spec)
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
        base = eng.run(prompt, EMPTY_EDITS, medians=medians,
                       max_new_tokens=max_new, seed=engine.SEED, cancel=_cancel)
        if _cancel.is_set():
            yield _event(stage="stopped", after="baseline")
            return

        yield _event(stage="steered")
        steered = eng.run(prompt, EditSet.from_rows(rows), medians=medians,
                          max_new_tokens=max_new, seed=engine.SEED, cancel=_cancel)
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
        key = steer.calibration_cache_key(
            engine._active_model_id, eng.revision, tok, str(steer.DTYPE), steer.DEVICE)
        with eng.clean():
            medians, cal_info = steer.calibrate_layer_medians(
                model, tok, calib_prompts, [layer],
                cache_path=ROOT / steer.calibration_path_for(engine._active_model_id),
                key=key, spec=spec)
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
