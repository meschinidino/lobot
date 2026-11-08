"""Routes: evidence. Helpers live in server_helpers. State lives in server_state.STATE."""
from __future__ import annotations

from fastapi import APIRouter
from model_lobotomy.server_state import STATE
from model_lobotomy import server_helpers as helpers
from model_lobotomy.server_schemas import (FollowupRequest)
from starlette.responses import FileResponse
from fastapi import HTTPException
from typing import Literal
from model_lobotomy import followups as followups_mod
from model_lobotomy import jobs_store
import json

router = APIRouter()


@router.get("/api/traits")
def traits(model_id: str | None = None):
    """Finished traits with evidence summaries (E1). No model needed."""
    items = helpers._trait_items()
    if model_id is not None:
        items = [t for t in items if t["model_id"] == model_id]
    return {"traits": items}

@router.get("/api/traits/{job_id}/report")
def trait_report(job_id: str):
    """One trait's report.json, plus pareto, repair and tuning.chosen (U4)."""
    trait_dir = jobs_store.JobPaths(job_id, helpers._trait_job_dir(job_id).parent).trait.workdir
    path = jobs_store.WorkPaths(trait_dir).report
    if not path.is_file():
        raise HTTPException(404, f"Unknown trait {job_id}.")
    rep = helpers._enrich_report(trait_dir, json.loads(path.read_text()))
    fu = STATE.get_followups()
    rep["followups"] = {k: fu.status(job_id, k, trait_dir)
                        for k in followups_mod.KINDS}
    return rep

@router.post("/api/traits/{job_id}/pareto")
def trait_pareto_start(job_id: str):
    """Run the PP stage for a finished Create (background job)."""
    return helpers._followup_start(job_id, "pareto", {})

@router.get("/api/traits/{job_id}/pareto")
def trait_pareto(job_id: str):
    return helpers._followup_status(job_id, "pareto")

@router.post("/api/traits/{job_id}/repair")
def trait_repair_start(job_id: str, req: FollowupRequest | None = None):
    """Run GR guard repair for a finished Create (background job)."""
    return helpers._followup_start(job_id, "repair",
                           {"method": (req or FollowupRequest()).method})

@router.get("/api/traits/{job_id}/repair")
def trait_repair(job_id: str):
    return helpers._followup_status(job_id, "repair")

@router.post("/api/traits/{job_id}/{kind}/stop")
def trait_followup_stop(job_id: str, kind: Literal["pareto", "repair"]):
    trait_dir = jobs_store.JobPaths(job_id, helpers._trait_job_dir(job_id).parent).trait.workdir
    return {"status": STATE.get_followups().stop(job_id, kind, trait_dir)}

@router.get("/api/traits/{job_id}/repaired/{method}")
def trait_repaired_patch(job_id: str, method: Literal["gradient", "project"]):
    """The repaired patch, when a repair chose a candidate."""
    path = jobs_store.JobPaths(job_id, helpers._trait_job_dir(job_id).parent).trait.repaired_patch(method)
    if not path.is_file():
        raise HTTPException(404, f"No repaired patch for {job_id}.")
    return FileResponse(path, media_type="application/zip",
                        filename=f"{job_id}-repaired-{method}.lobot")

@router.get("/api/traits/{job_id}/report.md")
def trait_report_md(job_id: str):
    """One trait's report.md as Markdown (E1)."""
    path = jobs_store.JobPaths(job_id, helpers._trait_job_dir(job_id).parent).trait.workdir / "report.md"
    if not path.is_file():
        raise HTTPException(404, f"Unknown trait {job_id}.")
    return FileResponse(path, media_type="text/markdown")

@router.get("/api/traits/{job_id}/learned-patch")
def trait_learned_patch(job_id: str):
    """One trait's learned.lobot multi-vector patch (ML v3)."""
    job_dir = helpers._trait_job_dir(job_id)
    path = job_dir / "trait" / "learned.lobot"
    if not path.is_file():
        raise HTTPException(404, f"Unknown trait {job_id}.")
    return FileResponse(path, media_type="application/zip",
                        filename=f"{job_id}-learned.lobot")


@router.get("/api/traits/{job_id}/patch")
def trait_patch(job_id: str):
    """One trait's knob.lobot patch as a download (E1)."""
    job_dir = helpers._trait_job_dir(job_id)
    path = jobs_store.WorkPaths(job_dir / "trait").patch
    if not path.is_file():
        raise HTTPException(404, f"Unknown trait {job_id}.")
    item = next((t for t in helpers._trait_items() if t["id"] == job_id), None)
    if item is not None:
        family, version = item["family"], item["version"]
    else:
        family, version = job_id, 1
        rep_path = jobs_store.WorkPaths(job_dir / "trait").report
        if rep_path.is_file():
            try:
                family = json.loads(rep_path.read_text()).get(
                    "trait", job_id)
            except ValueError:
                pass
    return FileResponse(path, media_type="application/zip",
                        filename=f"{family}-v{version}.lobot")
