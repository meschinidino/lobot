"""Routes: create. Helpers live in server_helpers. State lives in server_state.STATE."""
from __future__ import annotations

from fastapi import APIRouter
from model_lobotomy.server_state import STATE
from model_lobotomy import server_helpers as helpers
from model_lobotomy.server_schemas import (CreateRequest)
from fastapi import HTTPException
from model_lobotomy import app as engine
from model_lobotomy import trait_store

router = APIRouter()


@router.get("/api/families")
def families():
    runner = STATE.get_runner()
    return {"families": [
        {"name": f.name, "aliases": list(f.aliases),
         "pos": f.pos, "neg": f.neg, "judge": f.judge,
         "hand_validated": f.hand_validated,
         "validation": f.validation}
        for f in runner.store.all()]}

@router.post("/api/create")
def create(req: CreateRequest):
    if STATE.no_model:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    helpers._need_torch(helpers.get_engine(), "create")
    runner = STATE.get_runner()
    try:
        job_id = runner.start(req.description, seed=req.seed,
                              model_id=engine._active_model_id,
                              guard=req.guard,
                              research_mode=helpers._effective_research(req.research_mode),
                              schedule=req.schedule, pareto=req.pareto)
    except trait_store.UnsupportedTrait as e:
        raise HTTPException(422, str(e)) from None
    return {"job_id": job_id, **runner.status(job_id)}

@router.get("/api/create/{job_id}")
def create_status(job_id: str):
    try:
        return STATE.get_runner().reattach(job_id)
    except KeyError:
        raise HTTPException(404, f"Unknown job {job_id}.") from None

@router.post("/api/create/{job_id}/stop")
def create_stop(job_id: str):
    try:
        return STATE.get_runner().stop(job_id)
    except KeyError:
        raise HTTPException(404, f"Unknown job {job_id}.") from None

@router.get("/api/jobs")
def jobs():
    return {"jobs": STATE.get_runner().list_jobs()}
