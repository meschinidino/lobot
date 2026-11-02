"""Route: tune. Muse Spark picks doses; the local checks decide.

The guard stays on for every try and for the final check. The
request has no guard switch and no Research mode switch.
"""
from __future__ import annotations

import json
import queue
import threading
import time

from fastapi import APIRouter, HTTPException

from model_lobotomy import app as engine
from model_lobotomy import guard as guard_mod
from model_lobotomy import jobs_store
from model_lobotomy import patch as patch_mod
from model_lobotomy import server_helpers as helpers
from model_lobotomy import trait_eval
from model_lobotomy import tune as tune_mod
from model_lobotomy.server_schemas import TuneRequest
from model_lobotomy.server_state import STATE

router = APIRouter()

MAX_NEW_TOKENS = 60


@router.post("/api/tune")
def tune(req: TuneRequest):
    """Stream one event per tune step (NDJSON).

    Events: prediction, measured, stop, final, summary, done, and
    error. The final event holds the local verdict.
    """
    if STATE.replay:
        raise HTTPException(503, "Tune needs the model. Replay mode "
                                 "has no weights.")
    if STATE.no_model:
        raise HTTPException(503, "Weights are not loaded "
                                 "(MODEL_LOBOTOMY_NO_MODEL is set).")
    muse = helpers._muse()
    trait_dir = jobs_store.JobPaths(
        req.patch_id, helpers._trait_job_dir(req.patch_id).parent
    ).trait.workdir
    paths = jobs_store.WorkPaths(trait_dir)
    if not paths.report.is_file() or not paths.patch.is_file():
        raise HTTPException(404, f"No patch for {req.patch_id} yet.")
    rep = json.loads(paths.report.read_text())
    loaded = patch_mod.load_patch(paths.patch)
    meta = loaded["meta"]
    label = trait_eval.family_cfg(rep["trait"])["judge"]

    def gen():
        eng = helpers.get_engine()
        if not eng.lock.acquire(blocking=False):
            yield helpers._event(stage="error", message=(
                "A run is in progress. Wait for it to finish."))
            return
        try:
            if eng.model is None or eng.model_id != engine._active_model_id:
                eng.load(engine._active_model_id)
            helpers._need_torch(eng, "create")
            medians, _ = helpers._calibrate(eng, [int(meta["layer"])])
            yield from _tune_events(eng, muse, req, rep, loaded, label,
                                    medians)
        except Exception as e:  # noqa: BLE001 - reported to the client
            yield helpers._event(stage="error",
                                 message=f"{type(e).__name__}: {e}")
        finally:
            eng.lock.release()

    return helpers._ndjson(gen())


def _edits(loaded: dict, dose: float):
    from model_lobotomy import engine as engine_mod
    meta = loaded["meta"]
    unit = tuple(float(x) for x in loaded["direction"])
    return engine_mod.EditSet(vector=(engine_mod.VectorEdit(
        layer=int(meta["layer"]), direction=unit,
        scale=float(meta["scale"]), dose=float(dose)),))


def _score_prompts(eng, edits, prompts, label, medians):
    """(mean gain, pairs) of edited vs plain answers on prompts."""
    from model_lobotomy import engine as engine_mod
    pairs, gains = [], []
    for q in prompts:
        plain = eng.run(q, engine_mod.EMPTY_EDITS, medians=medians,
                        max_new_tokens=MAX_NEW_TOKENS,
                        seed=engine.SEED)["text"]
        edited = eng.run(q, edits, medians=medians,
                         max_new_tokens=MAX_NEW_TOKENS,
                         seed=engine.SEED)["text"]
        jb = trait_eval.judge_score(eng.model, eng.tokenizer, eng.spec,
                                    q, plain, label)
        js = trait_eval.judge_score(eng.model, eng.tokenizer, eng.spec,
                                    q, edited, label)
        gains.append(js - jb)
        pairs.append({"q": q, "plain": plain, "edited": edited,
                      "gain": round(js - jb, 3)})
    return sum(gains) / len(gains), pairs


def _tune_events(eng, muse, req, rep, loaded, label, medians):
    from model_lobotomy import guards as guards_mod
    splits = rep.get("splits") or {}
    tuning = (splits.get("tuning") or [])[:tune_mod.TUNING_PROMPTS]
    heldout = (splits.get("final") or [])[:tune_mod.HELDOUT_PROMPTS]
    if len(tuning) < tune_mod.TUNING_PROMPTS or \
            len(heldout) < tune_mod.HELDOUT_PROMPTS:
        yield helpers._event(stage="error", message=(
            "This report lacks 8 tuning or 8 held-out prompts."))
        return
    guard = guards_mod.get_guard()

    def try_fn(dose):
        t0 = time.time()
        edits = _edits(loaded, dose)
        screen = guard.screen(eng, edits, medians=medians)
        gain, pairs = _score_prompts(eng, edits, tuning, label, medians)
        return {"gain": gain,
                "guard_pass": not (screen.blocked or screen.would_block),
                "guard": screen.to_dict(), "pairs": pairs,
                "seconds": round(time.time() - t0, 1)}

    def final_fn(dose):
        t0 = time.time()
        edits = _edits(loaded, dose)
        full = guard.confirm(
            eng, edits, medians=medians, max_new_tokens=24,
            seed=engine.SEED, judge_fn=guard_mod.refusal_judge_fn(eng))
        gain, _pairs = _score_prompts(eng, edits, heldout, label, medians)
        return {"gain": gain, "guard": full.to_dict(),
                "seconds": round(time.time() - t0, 1)}

    events: queue.Queue = queue.Queue()
    box: dict = {}
    agent = tune_mod.TuneAgent(
        muse, try_fn, final_fn,
        patch_label=(f"{rep.get('trait')} patch at layer "
                     f"{loaded['meta']['layer']}, shipped dose "
                     f"{loaded['meta']['dose']}"),
        progress=lambda kind, row: events.put(
            helpers._event(stage=kind, **row)))

    def work():
        try:
            box["result"] = agent.run(req.goal, req.patch_id)
        except Exception as e:  # noqa: BLE001 - reported to the client
            box["error"] = e
        finally:
            events.put(None)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    while True:
        item = events.get()
        if item is None:
            break
        yield item
    thread.join()
    if "error" in box:
        yield helpers._event(stage="error", message=(
            f"{type(box['error']).__name__}: {box['error']}"))
        return
    result = box["result"]
    yield helpers._event(stage="done", result={
        "tune_id": result["tune_id"], "status": result["status"],
        "final": result["final"], "summary": result["summary"],
        "tries": [{k: v for k, v in t.items() if k != "pairs"}
                  for t in result["tries"]]})
