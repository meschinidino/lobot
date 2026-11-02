"""Routes: create. Helpers and state live in server (late import)."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, Query, Request, UploadFile  # noqa: F401

router = APIRouter()


class _Live:
    def __getattr__(self, name):
        from model_lobotomy import server as _server
        return getattr(_server, name)


_live = _Live()


@router.get("/api/families")
def families():
    runner = _get_runner()
    return {"families": [
        {"name": f.name, "aliases": list(f.aliases),
         "pos": f.pos, "neg": f.neg, "judge": f.judge,
         "hand_validated": f.hand_validated,
         "validation": f.validation}
        for f in runner.store.all()]}

@router.post("/api/create")
def create(req: CreateRequest):
    if _live.NO_MODEL:
        raise HTTPException(503, "Weights are not loaded (MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to run experiments.")
    _need_torch(get_engine(), "create")
    runner = _get_runner()
    try:
        job_id = runner.start(req.description, seed=req.seed,
                              model_id=engine._active_model_id,
                              guard=req.guard,
                              research_mode=req.research_mode,
                              schedule=req.schedule, pareto=req.pareto)
    except trait_store.UnsupportedTrait as e:
        raise HTTPException(422, str(e)) from None
    return {"job_id": job_id, **runner.status(job_id)}

@router.get("/api/create/{job_id}")
def create_status(job_id: str):
    try:
        return _get_runner().reattach(job_id)
    except KeyError:
        raise HTTPException(404, f"Unknown job {job_id}.") from None

@router.post("/api/create/{job_id}/stop")
def create_stop(job_id: str):
    try:
        return _get_runner().stop(job_id)
    except KeyError:
        raise HTTPException(404, f"Unknown job {job_id}.") from None

@router.get("/api/jobs")
def jobs():
    return {"jobs": _get_runner().list_jobs()}
