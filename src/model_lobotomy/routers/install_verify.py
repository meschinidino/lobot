"""Routes: install_verify. Helpers and state live in server (late import)."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, Query, Request, UploadFile  # noqa: F401

router = APIRouter()


class _Live:
    def __getattr__(self, name):
        from model_lobotomy import server as _server
        return getattr(_server, name)


_live = _Live()


@router.get("/api/install/targets")
def install_targets():
    """Probe llama.cpp and Ollama. Needs no weights (IN)."""
    return {"targets": [install_mod.probe(t)
                        for t in install_mod.TARGETS]}

@router.post("/api/install")
def install_patch(req: InstallRequest):
    """Export a trait patch and install it into one target (IN)."""
    if req.target not in install_mod.TARGETS:
        raise HTTPException(400, f"Unknown target {req.target!r}; "
                                 f"use one of {install_mod.TARGETS}.")
    if not req.model_ref.strip():
        raise HTTPException(400, "Give a model_ref.")
    job_dir = _trait_job_dir(req.trait_id)
    patch_path = jobs_store.WorkPaths(job_dir / "trait").patch
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

@router.post("/api/install/{install_id}/verify")
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

@router.post("/api/verify")
def verify(req: VerifyRequest):
    """Re-run a trait patch's shipped evidence (RV)."""
    job_dir = _trait_job_dir(req.trait_id)
    patch_path = jobs_store.WorkPaths(job_dir / "trait").patch
    if not patch_path.is_file():
        raise HTTPException(422, f"Trait {req.trait_id} has no saved "
                                 f"patch to verify.")
    if _live.NO_MODEL:
        raise HTTPException(503, "Weights are not loaded "
                                 "(MODEL_LOBOTOMY_NO_MODEL is set). "
                                 "Restart without it to verify.")
    if get_engine().lock.locked():
        raise HTTPException(409, "A run is in progress. "
                                 "Wait for it to finish.")
    _need_torch(get_engine(), "verify")
    _verify_pin(patch_path)
    _live._cancel.clear()
    return _ndjson(_verify_stream(patch_path, req.quick))

@router.post("/api/verify/upload")
async def verify_upload(file: UploadFile = File(...)):
    """Verify an uploaded .lobot file (RV). Never unpickles."""
    if _live.NO_MODEL:
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
    _live._cancel.clear()
    return _ndjson(_verify_stream(data, True))

@router.post("/api/combine/check")
def combine_check(req: CombineCheckRequest):
    """Pair matrix + install order for saved patches (CB)."""
    if _live.NO_MODEL:
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

@router.get("/api/combine/installed")
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
    if _live.NO_MODEL:
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
