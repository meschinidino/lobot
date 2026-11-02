"""Follow-up jobs on a finished Create run: Pareto (PP) and repair (GR).

pareto_job and repair_job hold the logic that scripts/pareto.py and
scripts/repair_patch.py used to hold, so the scripts and the server
share one code path. FollowupRunner runs them from the server in a
thread, one at a time on the engine lock (the same lock Create uses),
and persists status to <workdir>/<kind>.status.json so a page can
reattach.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path

from model_lobotomy import audit
from model_lobotomy import engine as engine_mod
from model_lobotomy import evaluation as E
from model_lobotomy import guard as guard_mod
from model_lobotomy import guard_screens as screens_mod
from model_lobotomy import pareto as pareto_mod
from model_lobotomy import patch as patch_mod
from model_lobotomy import repair as repair_mod
from model_lobotomy import trait_eval

GUARD_TOKENS = 24  # same as Create's guard and lobot_registry.py guard
KINDS = ("pareto", "repair")
REPAIR_METHODS = ("gradient", "project")


def repair_stem(method: str) -> str:
    """repair.json for project (gr-v1), repair_<method>.json otherwise."""
    return "repair" if method == "project" else f"repair_{method}"


def _no_progress(stage, done, total):
    return None


def pareto_job(eng, workdir, lane=None, *, max_new=None, cancel=None,
               progress=None) -> dict:
    """Score every PP point on the run's final split; write pareto.json."""
    workdir = Path(workdir)
    job = trait_eval.load_job(workdir)
    report, meta = job["report"], job["meta"]
    cfg = trait_eval.family_cfg(report["trait"])
    max_new = max_new or trait_eval.schedule_params(
        report.get("schedule", "full"))["gen_tokens"]
    lane = lane or trait_eval.TorchLane(eng, eng.spec)
    block = pareto_mod.run_pareto(
        lane, cfg["judge"], report["splits"]["final"],
        layer=int(meta["layer"]), unit=job["direction"].float(),
        scale=float(meta["scale"]), frozen_dose=float(meta["dose"]),
        instruction=report["prompt_baseline"]["top"], max_new=max_new,
        cancel=cancel, progress=progress)
    block["lane"] = lane.info()
    block["trait"] = report["trait"]
    (workdir / "pareto.json").write_text(json.dumps(block, indent=2))
    return block


def repair_job(eng, workdir, lane=None, *, method="gradient", cancel=None,
               progress=None, log=None) -> dict:
    """GR on a finished run; writes repair_<method>.json and maybe a patch.

    Progress counts steps: the confirmed guard, one tuning gain per
    candidate tried, and the chosen candidate's card and full guard.
    """
    if method not in REPAIR_METHODS:
        raise ValueError(f"unknown method {method!r}; have {REPAIR_METHODS}")
    workdir = Path(workdir)
    progress = progress or _no_progress
    log = log or (lambda msg: None)
    job = trait_eval.load_job(workdir)
    report, meta = job["report"], job["meta"]
    model_id = report["fingerprint"]["model_id"]
    cfg = trait_eval.family_cfg(report["trait"])
    params = trait_eval.schedule_params(report.get("schedule", "full"))
    seed = int(report.get("seed", 0))
    layer, scale = int(meta["layer"]), float(meta["scale"])
    splits = report["splits"]
    lane = lane or trait_eval.TorchLane(eng, eng.spec)
    base_cache = trait_eval.BaseCache()
    g = guard_mod.Guard()
    judge = guard_mod.refusal_judge_fn(eng)
    candidates = (repair_mod.GRADIENT_CANDIDATES if method == "gradient"
                  else repair_mod.CANDIDATES)
    total = 3 + len(candidates)
    steps = [0]

    def step():
        steps[0] += 1
        progress("repair", min(steps[0], total), total)

    def edits(unit, dose):
        return engine_mod.EditSet(vector=(engine_mod.VectorEdit(
            layer, tuple(unit.tolist()), scale, float(dose)),))

    def guard_fn(unit, dose, screens):
        trait_eval._check(cancel)
        ed, out = edits(unit, dose), {}
        for name in screens:
            if name == "refusal":
                v = g._refusal_entry(g.check_full(
                    eng, ed, medians={}, max_new_tokens=GUARD_TOKENS,
                    seed=seed, judge_fn=judge))[1]
            else:
                v = g.check_screen(eng, ed, name, medians={},
                                   max_new_tokens=GUARD_TOKENS, seed=seed)
            out[name] = screens_mod.screen_dict(
                name, v, guard_mod._SCREEN_DISPLAY[name])
        log(f"guard dose {dose:g}: {repair_mod.flagged(out) or 'clean'}")
        return out

    def confirm_fn(unit, dose):
        trait_eval._check(cancel)
        v = g.check_confirmed(eng, edits(unit, dose), medians={},
                              max_new_tokens=GUARD_TOKENS, seed=seed,
                              judge_fn=judge)
        verdict = {**guard_mod.verdict_dict(v), "enforced": False,
                   "research_mode": False, "confirmed": True}
        log(f"confirmed dose {dose:g}: would_block={v.would_block}")
        step()
        return {"screens": verdict.get("screens", {}), "verdict": verdict}

    def tuning_gain_fn(unit, dose):
        gains, _ = trait_eval._loop_gains(
            eng, eng.spec, cfg, splits["tuning"][:params["tune_n"]],
            edits(unit, dose), +1.0, params, base_cache, cancel, None,
            "tune", lane)
        log(f"tuning gain dose {dose:g}: {sum(gains) / len(gains):+.3f}")
        step()
        return gains

    def final_card_fn(unit, dose):
        shuffled = trait_eval._shuffled_unit(
            eng, eng.spec, cfg, splits["extraction"], layer, cancel=cancel)
        fin = trait_eval.verb_final(
            eng, eng.spec, cfg, cfg["judge"], splits["final"],
            edits(unit, dose), edits(shuffled, dose),
            n0=params["final_n0"], nmax=params["final_nmax"],
            max_new=params["gen_tokens"], base_cache=base_cache,
            cancel=cancel, lane=lane)
        card = trait_eval._json_card(
            E.report_card_pass(fin["gains"], fin["control"]))
        log(f"final card: passed={card['passed']} "
            f"mean={card['trait_mean']:+.3f}")
        step()
        return card

    unit0 = job["direction"].float() / job["direction"].float().norm()

    def safety_fn(screen):
        trait_eval._check(cancel)
        if method == "gradient":
            return repair_mod.cost_gradients(eng, screen, layer, unit0,
                                             float(meta["dose"]) * scale)
        return repair_mod.safety_direction(
            lambda msgs, L: repair_mod.capture_messages(eng, msgs, L),
            screen, layer)

    progress("repair", 0, total)
    rec = repair_mod.repair(
        job["direction"], float(meta["dose"]), safety_fn=safety_fn,
        guard_fn=guard_fn, tuning_gain_fn=tuning_gain_fn,
        confirm_fn=confirm_fn, final_card_fn=final_card_fn,
        candidates=candidates)
    out = {"trait": report["trait"], "model_id": model_id,
           "layer": layer, "lane": lane.info(), "method": method,
           "original_card": meta.get("report_card"),
           **repair_mod.record_json(rec)}
    if rec.get("chosen"):
        ch = rec["chosen"]
        recipe = {**(meta.get("recipe") or {}), "dose": ch["dose"],
                  "repair": {"version": repair_mod.REPAIR_VERSION,
                             "method": ch["method"],
                             "factor": ch["factor"],
                             "removed": rec["flagged"]
                             if ch["method"] == "project" else [],
                             "cos_to_original": ch["cos_to_original"],
                             "source_dose": float(meta["dose"])}}
        path = patch_mod.save_patch(
            workdir / f"knob.repaired-{method}.lobot",
            direction=rec["unit"],
            layer=layer, dose=ch["dose"], scale=scale,
            trait=report["trait"], recipe=recipe,
            report_card=ch["card"], guard=ch["after"]["verdict"],
            fingerprint=meta.get("fingerprint") or report["fingerprint"])
        out["patch"] = path.name
        audit.record("repair", path=str(path), status=rec["status"],
                     method=ch["method"], factor=ch["factor"])
    progress("repair", total, total)
    (workdir / f"{repair_stem(method)}.json").write_text(
        json.dumps(out, indent=2))
    return out


class FollowupRunner:
    """Runs Pareto and repair jobs for finished Creates, one at a time.

    The status of (job, kind) lives in memory while it runs and in
    <workdir>/<kind>.status.json for reattach. lane_fn(eng, model_id,
    revision) gives the Create lane (None means torch).
    """

    def __init__(self, engine, lane_fn=None, *, pareto_fn=None,
                 repair_fn=None):
        self._engine = engine
        self._lane_fn = lane_fn or (lambda eng, model_id, revision: None)
        self._fns = {"pareto": pareto_fn or pareto_job,
                     "repair": repair_fn or repair_job}
        self._mu = threading.Lock()
        self._live: dict[tuple, dict] = {}

    @staticmethod
    def _status_path(workdir: Path, kind: str) -> Path:
        return Path(workdir) / f"{kind}.status.json"

    def _persist(self, workdir: Path, st: dict) -> None:
        st["updated"] = time.time()
        self._status_path(workdir, st["kind"]).write_text(json.dumps(
            {k: v for k, v in st.items() if not k.startswith("_")},
            indent=2))

    def busy(self) -> dict | None:
        with self._mu:
            for st in self._live.values():
                if st["status"] in ("queued", "running"):
                    return st
        return None

    def start(self, job_id: str, kind: str, workdir, model_id: str,
              revision: str | None = None, **kw) -> dict:
        if kind not in KINDS:
            raise ValueError(f"unknown follow-up {kind!r}; have {KINDS}")
        workdir = Path(workdir)
        trait_eval.load_job(workdir)  # raises ValueError without a patch
        other = self.busy()
        if other is not None:
            raise RuntimeError(
                f"{other['kind']} for {other['job_id']} is running; "
                f"one follow-up at a time")
        st = {"job_id": job_id, "kind": kind, "status": "queued",
              "done": 0, "total": 0, "error": "", "options": kw,
              "started": time.time(), "_cancel": threading.Event()}
        with self._mu:
            self._live[(job_id, kind)] = st
        self._persist(workdir, st)
        threading.Thread(target=self._run,
                         args=(st, workdir, model_id, revision, kw),
                         daemon=True).start()
        return self.status(job_id, kind, workdir)

    def status(self, job_id: str, kind: str, workdir) -> dict | None:
        with self._mu:
            st = self._live.get((job_id, kind))
        if st is not None:
            return {k: v for k, v in st.items() if not k.startswith("_")}
        path = self._status_path(Path(workdir), kind)
        if path.is_file():
            disk = json.loads(path.read_text())
            if disk.get("status") in ("queued", "running"):
                # The process that ran it is gone.
                disk["status"] = "stopped"
                disk["error"] = "the server restarted during this job"
            return disk
        return None

    def stop(self, job_id: str, kind: str, workdir) -> dict | None:
        with self._mu:
            st = self._live.get((job_id, kind))
        if st is not None and st["status"] in ("queued", "running"):
            st["_cancel"].set()
        return self.status(job_id, kind, workdir)

    def _run(self, st, workdir, model_id, revision, kw):
        eng = self._engine
        lock = getattr(eng, "lock", None) or self._mu_run
        try:
            with lock:
                if st["_cancel"].is_set():
                    st["status"] = "stopped"
                    return
                st["status"] = "running"
                self._persist(workdir, st)
                if getattr(eng, "model", None) is None or getattr(
                        eng, "model_id", model_id) != model_id:
                    eng.load(model_id, revision)

                def progress(stage, done, total):
                    st["done"], st["total"] = done, total
                    self._persist(workdir, st)

                lane = self._lane_fn(eng, model_id,
                                     getattr(eng, "revision", revision))
                self._fns[st["kind"]](eng, workdir, lane,
                                      cancel=st["_cancel"],
                                      progress=progress, **kw)
                st["status"] = "done"
        except trait_eval.Stopped:
            st["status"] = "stopped"
        except Exception:  # noqa: BLE001 - recorded verbatim for the client
            st["status"] = "failed"
            st["error"] = traceback.format_exc()[-4000:]
        finally:
            self._persist(workdir, st)

    _mu_run = threading.Lock()
