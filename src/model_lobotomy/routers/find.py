"""Route: find. Muse Spark finds a neuron edit with lobot tools.

The guard stays on for every recipe try and for the final check.
The request has no guard switch and no Research mode switch.
"""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException

from model_lobotomy import app as engine
from model_lobotomy import find as find_mod
from model_lobotomy import guard as guard_mod
from model_lobotomy import pareto as pareto_mod
from model_lobotomy import reader as reader_mod
from model_lobotomy import trait_eval
from model_lobotomy import server_helpers as helpers
from model_lobotomy import steer
from model_lobotomy.server_schemas import FindRequest
from model_lobotomy.server_state import STATE

router = APIRouter()

MAX_NEW_TOKENS = 60


@router.post("/api/find-with-muse")
def find_with_claude(req: FindRequest):
    """Stream one event per find step (NDJSON).

    Events: prediction, result, stop, final, summary, done, error.
    """
    if STATE.replay:
        raise HTTPException(503, "Find needs the model. Replay mode "
                                 "has no weights.")
    if STATE.no_model:
        raise HTTPException(503, "Weights are not loaded "
                                 "(MODEL_LOBOTOMY_NO_MODEL is set).")
    muse = helpers._muse()

    def gen():
        eng = helpers.get_engine()
        if not eng.lock.acquire(blocking=False):
            yield helpers._event(stage="error", message=(
                "A run is in progress. Wait for it to finish."))
            return
        try:
            if eng.model is None or eng.model_id != engine._active_model_id:
                eng.load(engine._active_model_id)
            helpers._need_torch(eng, "reader")
            yield from _find_events(eng, muse, req)
        except Exception as e:  # noqa: BLE001 - reported to the client
            yield helpers._event(stage="error",
                                 message=f"{type(e).__name__}: {e}")
        finally:
            eng.lock.release()

    return helpers._ndjson(gen())


def _rows(args_rows):
    return [steer.RecipeRow(layer=int(r["layer"]),
                            neuron=int(r["neuron"]),
                            dose=float(r["dose"])) for r in args_rows]


def _fork_word(plain: str, edited: str):
    """First word where the edited answer leaves the plain one."""
    a, b = (plain or "").split(), (edited or "").split()
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return {"index": i, "plain": x, "edited": y}
    if len(b) > len(a):
        return {"index": len(a), "plain": None, "edited": b[len(a)]}
    return None


def _find_events(eng, muse, req):
    from model_lobotomy import guards as guards_mod
    from model_lobotomy.engine import EMPTY_EDITS, EditSet
    guard = guards_mod.get_guard()
    medians_box: dict = {}

    def medians(layers):
        key = tuple(sorted(set(layers)))
        if key not in medians_box:
            medians_box[key], _ = helpers._calibrate(eng, list(key))
        return medians_box[key]

    def find_neurons(args):
        got = reader_mod.find_neurons(
            eng, args["word"], None, k=min(int(args.get("k") or 10), 20))
        return {"word": got["word"], "results": got["results"][:10]}

    def read_neuron(args):
        layer, neuron = int(args["layer"]), int(args["neuron"])
        pushes = reader_mod.pushes_toward(eng, layer, neuron)
        cache_dir = (STATE.root / "store" / "reader"
                     / helpers._model_slug(engine._active_model_id))
        fires = None
        if (cache_dir / "fires_on.npz").exists():
            fires = reader_mod.fires_on(cache_dir, layer, neuron)
        return {"layer": layer, "neuron": neuron,
                "pushes_toward": pushes, "fires_on": fires}

    def measure_lift(args):
        layer = int(args["layer"])
        return reader_mod.steer_lift(
            eng, layer, int(args["neuron"]), float(args["dose"]),
            [str(w) for w in args["words"]],
            _lift_prompts(), medians([layer]))

    def edits_for(rows):
        return EditSet.from_rows(_rows(rows))

    def _fluency(prompt, text):
        return trait_eval.judge_score(
            eng.model, eng.tokenizer, eng.spec, prompt, text,
            pareto_mod.FLUENCY_LABEL)

    def try_recipe(args):
        rows = _rows(args["rows"])
        edits = edits_for(args["rows"])
        meds = medians([r.layer for r in rows] or [0])
        screen = guard.screen(eng, edits, medians=meds)
        plain = eng.run(req.prompt, EMPTY_EDITS, medians=meds,
                        max_new_tokens=MAX_NEW_TOKENS,
                        seed=engine.SEED)["text"]
        edited = eng.run(req.prompt, edits, medians=meds,
                         max_new_tokens=MAX_NEW_TOKENS,
                         seed=engine.SEED)["text"]
        return {"plain": plain, "edited": edited,
                "fluency": round(_fluency(req.prompt, edited), 3),
                "fork": _fork_word(plain, edited),
                "guard": screen.to_dict(),
                "guard_pass": not (screen.blocked or screen.would_block)}

    def final_fn(rows):
        if not rows:
            return {"guard": {"blocked": False, "would_block": False}}
        meds = medians([r["layer"] for r in rows])
        full = guard.confirm(
            eng, edits_for(rows), medians=meds, max_new_tokens=24,
            seed=engine.SEED, judge_fn=guard_mod.refusal_judge_fn(eng))
        return {"guard": full.to_dict()}

    def lift_fn(rows, words):
        meds = medians([r["layer"] for r in rows])
        return reader_mod.steer_lift_recipe(
            eng, _rows(rows), [str(w) for w in words],
            _lift_prompts(), meds)

    def fluency_fn(rows):
        meds = medians([r["layer"] for r in rows])
        edits = edits_for(rows)
        scores = []
        for q in _lift_prompts():
            edited = eng.run(q, edits, medians=meds,
                             max_new_tokens=MAX_NEW_TOKENS,
                             seed=engine.SEED)["text"]
            scores.append(_fluency(q, edited))
        return sum(scores) / len(scores)

    events: queue.Queue = queue.Queue()
    box: dict = {}
    agent = find_mod.FindAgent(
        muse,
        {"find_neurons": find_neurons, "read_neuron": read_neuron,
         "measure_lift": measure_lift, "try_recipe": try_recipe},
        final_fn, lift_fn, fluency_fn=fluency_fn,
        progress=lambda kind, row: events.put(
            helpers._event(stage=kind, **row)))

    def work():
        try:
            box["result"] = agent.run(req.goal, req.prompt)
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
        "find_id": result["find_id"], "status": result["status"],
        "tool_calls": result["tool_calls"], "final": result["final"],
        "summary": result["summary"], "muse": result.get("muse")})


def _lift_prompts() -> list:
    path = Path(STATE.root) / "store" / "reader" / "lift_prompts.json"
    return json.loads(path.read_text())
