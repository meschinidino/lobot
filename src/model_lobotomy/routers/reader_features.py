"""Routes: reader_features. Helpers and state live in server (late import)."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, Query, Request, UploadFile  # noqa: F401

router = APIRouter()


class _Live:
    def __getattr__(self, name):
        from model_lobotomy import server as _server
        return getattr(_server, name)


_live = _Live()


@router.post("/api/neurons")
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
    if _live.REPLAY:
        raise HTTPException(503, "Replay mode loads no weights, and the neurons at a word "
                                 "need the model. Restart without MODEL_LOBOTOMY_REPLAY.")
    if _live.NO_MODEL:
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

@router.get("/api/neuron/{layer}/{neuron}")
def neuron_card(layer: int, neuron: int, k: int = Query(12, ge=1, le=50)):
    """One neuron card: pushes_toward plus cached fires_on (R2)."""
    spec = engine.active_spec()
    if not 0 <= layer < spec.layers or not 0 <= neuron < spec.intermediate:
        raise HTTPException(400, f"Neuron L{layer}N{neuron} is out of "
                                 f"bounds [0,{spec.layers})x"
                                 f"[0,{spec.intermediate}).")
    cache_dir = _live.ROOT / "store" / "reader" / _model_slug(
        engine._active_model_id)
    if _live.REPLAY:
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
    if _live.NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    _need_torch(get_engine(), "reader")
    eng = get_engine()
    if not eng.lock.acquire(blocking=False):
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    try:
        if eng.model is None or eng.model_id != engine._active_model_id:
            eng.load(engine._active_model_id)
        pushes = _live.reader.pushes_toward(eng, layer, neuron, k=k)
        fires, built, tokens = None, False, 0
        if ((cache_dir / "fires_on.npz").exists()
                and (cache_dir / "sample.json").exists()):
            sample = json.loads((cache_dir / "sample.json").read_text())
            tokens = len(sample.get("tokens", []))
            fires = _live.reader.fires_on(cache_dir, layer, neuron, k=k)
            built = True
        return {"model_id": engine._active_model_id, "layer": layer,
                "neuron": neuron, "pushes_toward": pushes,
                "fires_on": fires,
                "fires_on_cache": {"built": built, "tokens": tokens}}
    finally:
        eng.lock.release()

@router.get("/api/find")
def find(word: str = "", layers: str | None = None,
         k: int = Query(20, ge=1, le=50)):
    """Neurons that push one word hardest (R2)."""
    spec = engine.active_spec()
    layer_list = _parse_layers(layers, spec)
    if not word or not word.strip():
        raise HTTPException(400, "The word is empty. Type a word first.")
    if _live.REPLAY:
        raise HTTPException(503, "Find needs the model. Replay mode has no recordings for it.")
    if _live.NO_MODEL:
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
            return _live.reader.find_neurons(eng, word, layer_list, k=k)
        except ValueError as e:
            raise HTTPException(400, f"{e}.") from None
    finally:
        eng.lock.release()

@router.post("/api/lift")
def lift(req: LiftRequest):
    """Steering lift for one neuron (R2)."""
    spec = engine.active_spec()
    if not 0 <= req.layer < spec.layers or not 0 <= req.neuron < spec.intermediate:
        raise HTTPException(400, f"Neuron L{req.layer}N{req.neuron} is out of "
                                 f"bounds [0,{spec.layers})x"
                                 f"[0,{spec.intermediate}).")
    prompts = req.prompts
    if prompts is None:
        path = _live.ROOT / "store" / "reader" / "lift_prompts.json"
        if not path.exists():
            raise HTTPException(500, "The lift prompts are missing from the server.")
        prompts = json.loads(path.read_text())
    if _live.REPLAY:
        raise HTTPException(503, "Lift needs the model. Replay mode has no recordings for it.")
    if _live.NO_MODEL:
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
            return _live.reader.steer_lift(eng, req.layer, req.neuron, req.dose,
                                     req.words, prompts, medians)
        except ValueError as e:
            raise HTTPException(400, f"{e}.") from None
    finally:
        eng.lock.release()

@router.get("/api/features")
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

@router.get("/api/feature/{layer}/{index}")
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
    if _live.REPLAY:
        raise HTTPException(503, "Feature cards need the model. "
                                 "Replay mode has no recordings for them.")
    if _live.NO_MODEL:
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

@router.post("/api/feature-lift")
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
        path = _live.ROOT / "store" / "reader" / "lift_prompts.json"
        if not path.exists():
            raise HTTPException(500, "The lift prompts are missing from the server.")
        prompts = json.loads(path.read_text())
    if _live.REPLAY:
        raise HTTPException(503, "Feature lift needs the model. "
                                 "Replay mode has no recordings for it.")
    if _live.NO_MODEL:
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
            words = sae_mod.top_words(_live.reader.pushes_of_direction(
                eng, sae_mod.feature_direction(sae, req.index),
                layer=req.layer))
        if not words:
            raise HTTPException(422, "Give at least one word.")
        medians, _ = _calibrate(eng, [req.layer])
        try:
            return _live.reader.steer_lift_vector(
                eng, req.layer, sae_mod.feature_direction(sae, req.index),
                req.dose, words, prompts, medians)
        except ValueError as e:
            raise HTTPException(400, f"{e}.") from None
    finally:
        eng.lock.release()
