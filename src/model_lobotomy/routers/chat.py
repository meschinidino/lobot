"""Routes: chat. Helpers and state live in server (late import)."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, Query, Request, UploadFile  # noqa: F401

router = APIRouter()


class _Live:
    def __getattr__(self, name):
        from model_lobotomy import server as _server
        return getattr(_server, name)


_live = _Live()


@router.post("/api/chat/start")
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

@router.get("/api/chat/{session_id}")
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

@router.get("/api/chat/{session_id}/transcript")
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

@router.post("/api/chat/replay")
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
    if _live.NO_MODEL:
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
        _live._cancel.clear()
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

@router.post("/api/chat")
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
    if _live.NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    if get_engine().lock.locked():
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    if sess.get("dial_traits"):
        _need_torch(get_engine(), "dial")
    _live._cancel.clear()
    max_new = sess["max_new_tokens"]
    if req.max_new_tokens is not None:
        max_new = min(req.max_new_tokens, steer.MAX_NEW_TOKENS_CAP)
    return _ndjson(_chat_stream(req.session_id, message, rows, max_new))
