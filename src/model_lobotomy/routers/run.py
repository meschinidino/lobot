"""Routes: run. Helpers and state live in server (late import)."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, Query, Request, UploadFile  # noqa: F401

router = APIRouter()


class _Live:
    def __getattr__(self, name):
        from model_lobotomy import server as _server
        return getattr(_server, name)


_live = _Live()


@router.get("/api/state")
def state():
    return _state()

@router.post("/api/model")
def switch_model(req: ModelRequest):
    if req.model_id not in engine._PRESETS:
        raise HTTPException(400, f"Unknown model {req.model_id}. "
                                 f"Supported: {', '.join(sorted(engine._PRESETS))}.")
    if get_engine().lock.locked():
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    engine._active_model_id = req.model_id
    engine.unload_engine()
    return _state()

@router.post("/api/run")
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
    if _live.REPLAY:
        rec = _find_recording(engine._active_model_id, prompt, [r.model_dump() for r in req.rows])
        if rec is None:
            raise HTTPException(404, "Replay mode answers only recorded setups, and this one was "
                                     "not recorded. Record it with scripts/record_sweep.py, or "
                                     "restart without MODEL_LOBOTOMY_REPLAY to run the model.")
        if get_engine().lock.locked():
            raise HTTPException(409, "A run is in progress. Wait for it to finish.")
        _live._cancel.clear()
        return _ndjson(_replay_stream(rec))
    if _live.NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    if req.dial_trait is not None:
        _need_torch(get_engine(), "dial")
    if get_engine().lock.locked():
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    _live._cancel.clear()
    return _ndjson(_run_stream(prompt, rows, spec, guard=req.guard,
                    research_mode=req.research_mode,
                    dial_trait=req.dial_trait))

@router.post("/api/stop")
def stop():
    _live._cancel.set()
    return {"stopping": get_engine().lock.locked()}

@router.post("/api/update")
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
    if _live.NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    if get_engine().lock.locked():
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    _live._cancel.clear()
    return _ndjson(_update_stream(prompt, layer, direction, scale, list(req.doses),
                       min(req.max_new_tokens, steer.MAX_NEW_TOKENS_CAP),
                       spec, guard=req.guard,
                       research_mode=req.research_mode))

@router.post("/api/delete")
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
    if _live.NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    _need_torch(get_engine(), "ablate")
    if get_engine().lock.locked():
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    _live._cancel.clear()
    return _ndjson(_delete_stream(prompt, req.trait, knob,
                       min(req.max_new_tokens, steer.MAX_NEW_TOKENS_CAP),
                       guard=req.guard,
                       research_mode=req.research_mode))

@router.post("/api/delete/check")
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
    if _live.NO_MODEL:
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
                prompts=prompts, cancel=_live._cancel)
        except ValueError as e:
            raise HTTPException(400, f"{e}.") from None
        out["seconds"] = round(time.perf_counter() - t_all, 1)
        return out
    finally:
        eng.lock.release()
