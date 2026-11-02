"""Routes: drafts_agent. Helpers live in server_helpers. State lives in server_state.STATE."""
from __future__ import annotations

from fastapi import APIRouter
from model_lobotomy.server_state import STATE
from model_lobotomy import server_helpers as helpers
from model_lobotomy.server_schemas import (AgentRequest, DraftRequest, LabelsRequest)
from fastapi import HTTPException
from model_lobotomy import drafts as drafts_mod
from model_lobotomy import app as engine
from model_lobotomy import guard as guard_mod
import hashlib
from model_lobotomy import jobs_store
import json
from model_lobotomy import prompt_bank
import threading
import time
from model_lobotomy import trait_eval
from model_lobotomy import trait_store
import uuid

router = APIRouter()


@router.post("/api/drafts")
def draft_create(req: DraftRequest):
    """Draft model-written parts + 20 candidate replies (C3)."""
    allowed, reason = guard_mod.draft_description_allowed(
        req.description)
    if not allowed:
        raise HTTPException(422, reason)
    if STATE.replay:
        raise HTTPException(503, "Drafts need the model. "
                                 "Replay mode has no recordings for them.")
    if STATE.no_model:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    eng = helpers.get_engine()
    if not eng.lock.acquire(blocking=False):
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    try:
        if eng.model is None or eng.model_id != engine._active_model_id:
            eng.load(engine._active_model_id)
        try:
            if req.drafter == "muse":
                parts = helpers._muse().draft_parts(req.description)
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
        draft_id = drafts_mod.new_draft_id(STATE.root / "store",
                                           req.description)
        record = drafts_mod.build_draft_record(
            draft_id, req.description, parts, eng.model_id,
            eng.revision)
        drafts_mod.save_draft(STATE.root / "store", draft_id, record,
                              replies)
        return {"id": draft_id, "description": record["description"],
                "parts": {k: record[k] for k in
                          ("pos", "neg", "judge", "baselines")},
                "replies": [drafts_mod.public_reply(r) for r in replies],
                "labeled": 0, "total": len(replies)}
    finally:
        eng.lock.release()

@router.post("/api/drafts/{draft_id}/prelabel")
def draft_prelabel(draft_id: str):
    """Muse Spark label suggestions; the user still saves the labels."""
    record, replies, labels = helpers._draft_or_404(draft_id)
    got = helpers._muse_call(helpers._muse().prelabel, record["description"], replies)
    return {"id": draft_id, "suggestions": got, "total": len(replies),
            "suggested": len(got), "labeled": len(labels)}

@router.post("/api/traits/{job_id}/explain")
def trait_explain(job_id: str):
    """Plain-words explanation of a finished report (Muse Spark).

    Cached in the job dir; the numbers stay the report's own.
    """
    trait_dir = jobs_store.JobPaths(job_id, helpers._trait_job_dir(job_id).parent).trait.workdir
    report = jobs_store.WorkPaths(trait_dir).report
    if not report.is_file():
        raise HTTPException(404, f"No report for {job_id} yet.")
    rep = helpers._enrich_report(trait_dir, json.loads(report.read_text()))
    # The cache is keyed by the numbers Muse sees, so a later Pareto or
    # repair run gets a fresh explanation.
    from model_lobotomy import muse as muse_mod
    key = hashlib.sha256(json.dumps(muse_mod.report_summary(rep),
                                    sort_keys=True).encode()).hexdigest()
    cached, key_path = trait_dir / "explanation.txt", trait_dir / "explanation.key"
    if cached.is_file() and key_path.is_file() and \
            key_path.read_text().strip() == key:
        return {"id": job_id, "explanation": cached.read_text().strip(),
                "cached": True}
    text = helpers._muse_call(helpers._muse().explain, rep)
    cached.write_text(text + "\n")
    key_path.write_text(key + "\n")
    return {"id": job_id, "explanation": text, "cached": False}

@router.post("/api/traits/{job_id}/second-opinion")
def trait_second_opinion(job_id: str):
    """Muse Spark re-scores the turn-up pairs (second opinion).

    The local judge still gates Create. This route changes no gate
    and no verdict. It caches by a hash of the pairs.
    """
    from model_lobotomy import second_opinion as so_mod
    trait_dir = jobs_store.JobPaths(job_id, helpers._trait_job_dir(job_id).parent).trait.workdir
    paths = jobs_store.WorkPaths(trait_dir)
    if not paths.report.is_file():
        raise HTTPException(404, f"No report for {job_id} yet.")
    rep = json.loads(paths.report.read_text())
    pairs = (((rep.get("final") or {}).get("update_up") or {})
             .get("pairs") or [])
    if not pairs:
        raise HTTPException(422, "This report has no turn-up pairs "
                                 "to re-score.")
    key = so_mod.pairs_hash(pairs)
    cached = helpers._read_json(paths.second_opinion)
    if cached and cached.get("pairs_hash") == key:
        return {"id": job_id, **cached["result"], "cached": True}
    try:
        label = trait_eval.family_cfg(rep.get("trait"))["judge"]
    except KeyError:
        raise HTTPException(422, f"No judge label for "
                                 f"{rep.get('trait')!r}.") from None

    def factory():
        try:
            return trait_eval.MetaApiJudge()
        except trait_eval.MetaApiError as e:
            raise HTTPException(503, f"Muse Spark is not available: "
                                     f"{e}. Set META_API_KEY.") from None

    try:
        factory().model  # fail fast with 503 before any scoring
    except HTTPException:
        raise
    try:
        result = so_mod.run_second_opinion(pairs, label, factory)
    except trait_eval.MetaApiError as e:
        raise HTTPException(502, f"Muse Spark failed: {e}") from None
    paths.second_opinion.write_text(json.dumps(
        {"pairs_hash": key, "result": result}, indent=1))
    return {"id": job_id, **result, "cached": False}

@router.post("/api/agent")
def agent_start(req: AgentRequest):
    """Start the Create agent (Muse Spark + local gates) in the background."""
    allowed, reason = guard_mod.draft_description_allowed(req.description)
    if not allowed:
        raise HTTPException(422, reason)
    if STATE.replay or STATE.no_model:
        raise HTTPException(503, "The agent needs the model loaded.")
    helpers._need_torch(helpers.get_engine(), "create")
    muse = helpers._muse()
    if not STATE.agent_lock.acquire(blocking=False):
        raise HTTPException(409, "An agent run is in progress.")
    eng = helpers.get_engine()
    if not eng.lock.acquire(blocking=False):
        STATE.agent_lock.release()
        raise HTTPException(409, "A run is in progress. Wait for it to finish.")
    from model_lobotomy import agent as agent_mod
    cancel = threading.Event()
    state = {"status": "running", "steps": [], "result": None,
             "error": None, "cancel": cancel,
             "description": req.description, "started": time.time(),
             "job_id": None}

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
                eng, muse, store=STATE.get_runner().store, lane=lane,
                max_rounds=req.max_rounds, cancel=cancel,
                progress=progress)
            state["id"] = ag.id
            state["result"] = ag.run(req.description)
            state["status"] = state["result"]["status"]
            state["job_id"] = helpers._register_agent_job(state["result"])
        except agent_mod.AgentStopped:
            state["status"] = "stopped"
        except Exception as e:  # noqa: BLE001 - reported to the client
            state["status"] = "failed"
            state["error"] = f"{type(e).__name__}: {e}"[:2000]
        finally:
            eng.lock.release()
            STATE.agent_lock.release()

    run_id = uuid.uuid4().hex[:12]
    STATE.agents[run_id] = state
    threading.Thread(target=work, daemon=True).start()
    return {"run_id": run_id, "status": "running"}

@router.get("/api/agents")
def agents_list():
    """Agent runs: live ones in memory, finished ones from disk."""
    out, seen = [], set()
    for run_id, st in STATE.agents.items():
        seen.add(st.get("id"))
        out.append({"run_id": run_id, "id": st.get("id"),
                    "status": st["status"],
                    "description": (st.get("result") or {}).get(
                        "description") or st.get("description"),
                    "family": (st.get("result") or {}).get("family"),
                    "job_id": st.get("job_id"),
                    "started": st.get("started")})
    root = helpers._agents_dir()
    if root.is_dir():
        for d in sorted(root.iterdir(), reverse=True):
            if d.name in seen:
                continue
            st = helpers._disk_agent(d.name)
            if st is None:
                continue
            out.append({"run_id": d.name, "id": d.name,
                        "status": st["status"],
                        "description": st["result"].get("description"),
                        "family": st["result"].get("family"),
                        "job_id": st["job_id"], "started": None})
    return {"agents": out}

@router.get("/api/agent/{run_id}")
def agent_status(run_id: str):
    state = STATE.agents.get(run_id)
    if state is None:
        disk = helpers._disk_agent(run_id)
        if disk is None:
            raise HTTPException(404, f"Unknown agent run {run_id}.")
        return disk
    return {k: v for k, v in state.items() if k != "cancel"}

@router.post("/api/agent/{run_id}/stop")
def agent_stop(run_id: str):
    state = STATE.agents.get(run_id)
    if state is None:
        raise HTTPException(404, f"Unknown agent run {run_id}.")
    state["cancel"].set()
    return {"run_id": run_id, "stopping": True}

@router.get("/api/drafts/{draft_id}")
def draft_get(draft_id: str):
    """One draft: parts + labeling queue with labels so far (C3)."""
    record, replies, labels = helpers._draft_or_404(draft_id)
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

@router.post("/api/drafts/{draft_id}/labels")
def draft_label(draft_id: str, req: LabelsRequest):
    """Submit user labels for candidate replies (C3)."""
    _record, replies, labels = helpers._draft_or_404(draft_id)
    reply_ids = {r["reply_id"] for r in replies}
    try:
        merged = drafts_mod.merge_labels(
            labels, [row.model_dump() for row in req.labels],
            reply_ids)
    except drafts_mod.DraftError as e:
        raise HTTPException(400, str(e)) from None
    drafts_mod.save_labels(STATE.root / "store", draft_id, merged)
    return {"id": draft_id, "labeled": len(merged),
            "total": len(replies)}

@router.post("/api/drafts/{draft_id}/validate")
def draft_validate(draft_id: str):
    """Judge-vs-user agreement; registers the family on pass (C3)."""
    record, replies, labels = helpers._draft_or_404(draft_id)
    if len(labels) < len(replies):
        raise HTTPException(
            422, f"Label all {len(replies)} replies first "
                 f"({len(labels)} labeled).")
    if STATE.replay:
        raise HTTPException(503, "Validation needs the model. "
                                 "Replay mode has no recordings for it.")
    if STATE.no_model:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    eng = helpers.get_engine()
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
        runner = STATE.get_runner()
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
