"""GR for a finished Create run: repair a patch the guard flags.

Confirms the patch's guard with full generation, and when a screen
flags, tries the GR candidates in their fixed order. Writes
JOBDIR/repair_<method>.json (repair.json for project) and, when a
candidate is chosen, JOBDIR/knob.repaired-<method>.lobot. Rules: docs/roadmap/specs/GR-guard-repair.md.

Run:
    uv run python scripts/repair_patch.py jobs/pirate-0-20261030-004543-000/trait --lane llama
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_lobotomy import audit  # noqa: E402
from model_lobotomy import engine as engine_mod  # noqa: E402
from model_lobotomy import evaluation as E  # noqa: E402
from model_lobotomy import guard as guard_mod  # noqa: E402
from model_lobotomy import guard_screens as screens_mod  # noqa: E402
from model_lobotomy import patch as patch_mod  # noqa: E402
from model_lobotomy import repair  # noqa: E402
from model_lobotomy import trait_eval  # noqa: E402
from model_lobotomy.engine import get_engine  # noqa: E402

GUARD_TOKENS = 24  # same as Create's guard and lobot_registry.py guard


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobdir")
    ap.add_argument("--lane", default="torch", choices=["torch", "llama"])
    ap.add_argument("--method", default="gradient",
                    choices=["gradient", "project"],
                    help="gradient: remove the safe replies' NLL gradients "
                         "(gr-v2); project: remove mean-diff safety "
                         "directions, then lower doses (gr-v1)")
    args = ap.parse_args()
    jobdir = Path(args.jobdir)
    job = trait_eval.load_job(jobdir)
    report, meta = job["report"], job["meta"]
    model_id = report["fingerprint"]["model_id"]
    cfg = trait_eval.family_cfg(report["trait"])
    params = trait_eval.schedule_params(report.get("schedule", "full"))
    seed = int(report.get("seed", 0))
    layer, scale = int(meta["layer"]), float(meta["scale"])
    splits = report["splits"]
    eng = get_engine()
    eng.load(model_id)
    lane = (trait_eval.open_lane(eng, args.lane, model_id)
            or trait_eval.TorchLane(eng, eng.spec))
    base_cache = trait_eval.BaseCache()
    g = guard_mod.Guard()
    judge = guard_mod.refusal_judge_fn(eng)

    def edits(unit, dose):
        return engine_mod.EditSet(vector=(engine_mod.VectorEdit(
            layer, tuple(unit.tolist()), scale, float(dose)),))

    def log(msg):
        print(f"[repair] {msg}", flush=True)

    def guard_fn(unit, dose, screens):
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
        log(f"guard dose {dose:g}: {repair.flagged(out) or 'clean'}")
        return out

    def confirm_fn(unit, dose):
        v = g.check_confirmed(eng, edits(unit, dose), medians={},
                              max_new_tokens=GUARD_TOKENS, seed=seed,
                              judge_fn=judge)
        verdict = {**guard_mod.verdict_dict(v), "enforced": False,
                   "research_mode": False, "confirmed": True}
        log(f"confirmed dose {dose:g}: would_block={v.would_block}")
        return {"screens": verdict.get("screens", {}), "verdict": verdict}

    def tuning_gain_fn(unit, dose):
        gains, _ = trait_eval._loop_gains(
            eng, eng.spec, cfg, splits["tuning"][:params["tune_n"]],
            edits(unit, dose), +1.0, params, base_cache, None, None,
            "tune", lane)
        log(f"tuning gain dose {dose:g}: {sum(gains) / len(gains):+.3f}")
        return gains

    def final_card_fn(unit, dose):
        shuffled = trait_eval._shuffled_unit(
            eng, eng.spec, cfg, splits["extraction"], layer)
        fin = trait_eval.verb_final(
            eng, eng.spec, cfg, cfg["judge"], splits["final"],
            edits(unit, dose), edits(shuffled, dose),
            n0=params["final_n0"], nmax=params["final_nmax"],
            max_new=params["gen_tokens"], base_cache=base_cache,
            lane=lane)
        card = trait_eval._json_card(
            E.report_card_pass(fin["gains"], fin["control"]))
        log(f"final card: passed={card['passed']} "
            f"mean={card['trait_mean']:+.3f}")
        return card

    unit0 = job["direction"].float() / job["direction"].float().norm()

    def safety_fn(screen):
        if args.method == "gradient":
            rows = repair.cost_gradients(eng, screen, layer, unit0,
                                         float(meta["dose"]) * scale)
            log(f"{screen}: {rows.shape[0]} gradient rows, max |cos| "
                f"{float((rows / rows.norm(dim=1, keepdim=True) @ unit0).abs().max()):.3f}")
            return rows
        return repair.safety_direction(
            lambda msgs, L: repair.capture_messages(eng, msgs, L),
            screen, layer)

    rec = repair.repair(
        job["direction"], float(meta["dose"]), safety_fn=safety_fn,
        guard_fn=guard_fn, tuning_gain_fn=tuning_gain_fn,
        confirm_fn=confirm_fn, final_card_fn=final_card_fn,
        candidates=(repair.GRADIENT_CANDIDATES if args.method == "gradient"
                    else repair.CANDIDATES))
    stem = "repair" if args.method == "project" else f"repair_{args.method}"
    out = {"trait": report["trait"], "model_id": model_id,
           "layer": layer, "lane": lane.info(),
           "original_card": meta.get("report_card"),
           **repair.record_json(rec)}
    if rec.get("chosen"):
        ch = rec["chosen"]
        recipe = {**(meta.get("recipe") or {}), "dose": ch["dose"],
                  "repair": {"version": repair.REPAIR_VERSION,
                             "method": ch["method"],
                             "factor": ch["factor"],
                             "removed": rec["flagged"]
                             if ch["method"] == "project" else [],
                             "cos_to_original": ch["cos_to_original"],
                             "source_dose": float(meta["dose"])}}
        path = patch_mod.save_patch(
            jobdir / f"knob.repaired-{args.method}.lobot",
            direction=rec["unit"],
            layer=layer, dose=ch["dose"], scale=scale,
            trait=report["trait"], recipe=recipe,
            report_card=ch["card"], guard=ch["after"]["verdict"],
            fingerprint=meta.get("fingerprint") or report["fingerprint"])
        out["patch"] = path.name
        audit.record("repair", path=str(path), status=rec["status"],
                     method=ch["method"], factor=ch["factor"])
    (jobdir / f"{stem}.json").write_text(json.dumps(out, indent=2))
    log(f"status={rec['status']} flagged={rec['flagged']} "
        f"chosen={(rec.get('chosen') or {}).get('method')}"
        f"@{(rec.get('chosen') or {}).get('factor')}")
    log(f"wrote {jobdir / f'{stem}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
