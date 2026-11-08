"""ML step 2 live search: combined-layer steering for one family.

Usage: uv run python scripts/combine_search.py <family> <job-dir>
[--model meta-llama/Llama-3.2-3B-Instruct]

Reads splits from the job's trait/report.json. Ranks every layer by
steer_score (ML step 1), searches mixes on the first 10 tuning
questions, confirms the winner on all 30 final questions, and writes
combine.json next to the report. Gates: docs/roadmap/specs/
ML-multi-layer-search.md. Take the model lane before running.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model_lobotomy import app as app_mod
from model_lobotomy import directions, engine as engine_mod
from model_lobotomy import layer_search as combine, pareto, steer, trait_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("job")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--v2", action="store_true",
                    help="probe ranking, harder push, brand-new confirm")
    a = ap.parse_args()
    job = ROOT / a.job if not Path(a.job).is_absolute() else Path(a.job)
    report = json.loads((job / "trait" / "report.json").read_text())
    cfg = trait_eval.family_cfg(a.family)
    search_prompts = report["splits"]["tuning"][:10]
    if a.v2:
        fresh_file = job / "trait" / "fresh_questions_v2.json"
        fresh_prompts = json.loads(fresh_file.read_text())["questions"]
    else:
        fresh_prompts = report["splits"]["final"]

    app_mod._active_model_id = a.model
    preset, spec = app_mod.active_preset(), app_mod.active_spec()
    eng = engine_mod.Engine()
    eng.load(a.model)
    judge = trait_eval.LocalJudge(eng.model, eng.tokenizer, spec)
    port = directions.get_direction()
    A, B = port.derive(eng, spec, cfg, report["splits"]["extraction"])
    seps = port.seps(A, B, list(range(spec.layers)))
    ranked = sorted(seps, key=lambda L: seps[L]["steer_score"],
                    reverse=True)
    print("LAYER RANKING (steer_score, d_prime, consistency, acc):")
    for L in ranked:
        s = seps[L]
        print(f"  L{L}: {s['steer_score']:.3f} {s['d_prime']:.3f} "
              f"{s['consistency']:.3f} {s['acc']:.3f}", flush=True)
    top = ranked[:combine.TOP_K]
    key = steer.calibration_cache_key(a.model, eng.revision,
                                      eng.tokenizer, str(steer.DTYPE),
                                      steer.DEVICE)
    medians, _ = steer.calibrate_layer_medians(
        eng.model, eng.tokenizer, preset["calibration_prompts"],
        list(range(spec.layers)) if a.v2 else top,
        cache_path=ROOT / steer.calibration_path_for(a.model),
        key=key, spec=spec)

    base_cache = {}

    def base_for(prompts):
        k = id(prompts)
        if k not in base_cache:
            rows = []
            for p in prompts:
                r = eng.run_batched(p, [engine_mod.EditSet()],
                                    medians=medians, max_new_tokens=100,
                                    seed=app_mod.SEED)
                rows.append((r["rows"][0]["text"],
                             judge.score(p, r["rows"][0]["text"],
                                         cfg["judge"])))
            base_cache[k] = rows
        return base_cache[k]

    score_cache = {}

    def score(rows, prompts):
        ck = (rows, id(prompts))
        if ck in score_cache:
            return score_cache[ck]
        edit = engine_mod.EditSet(vector=tuple(
            engine_mod.VectorEdit(L, tuple(seps[L]["unit"]),
                                  float(medians[L]), d) for L, d in rows))
        gains, flus = [], []
        for p, (_, jb) in zip(prompts, base_for(prompts)):
            r = eng.run_batched(p, [edit], medians=medians,
                                max_new_tokens=100, seed=app_mod.SEED)
            text = r["rows"][0]["text"]
            gains.append(judge.score(p, text, cfg["judge"]) - jb)
            flus.append(judge.score(p, text, pareto.FLUENCY_LABEL))
        out = {"gain": round(sum(gains) / len(gains), 3),
               "fluency_mean": round(sum(flus) / len(flus), 3),
               "fluency_min": round(min(flus), 3),
               "fluency_p10": round(trait_eval._p10(flus), 3)}
        score_cache[ck] = out
        return out

    probe = None
    if a.v2:
        ranked, probe = combine.probe_layers(
            score, list(range(spec.layers)), search_prompts[:5])
        print("PROBE RANKING (layer, probe_value, gain, fluency_p10):")
        for L in ranked:
            t = probe[L]
            print(f"  L{L}: {t['probe_value']:.3f} {t['gain']:.3f} "
                  f"{t['fluency_p10']:.3f}", flush=True)
    result = combine.greedy_search(score, ranked, search_prompts)
    print("SEARCH STEPS:")
    for s in result["steps"]:
        print(" ", s, flush=True)
    verdict = combine.confirm(score, result["rows"], fresh_prompts)
    print("BEST MIX:", result["rows"], "search:", result["search"])
    print("CONFIRM:", verdict, flush=True)
    eng.unload()
    out = {"family": a.family, "model": a.model,
           "ranked_layers": [{"layer": L,
                              **{k: round(seps[L][k], 3) for k in
                                 ("steer_score", "d_prime",
                                  "consistency", "acc")}}
                             for L in ranked],
           "probe": ({str(L): probe[L] for L in probe}
                     if probe else None),
           "search_prompts": len(search_prompts),
           "fresh_prompts": len(fresh_prompts),
           # The v1 final split also served the Create pipeline
           # before this search; only the search never used it.
           "fresh_set": ("fresh_questions_v2.json" if a.v2
                         else "report splits.final (Create used it first)"),
           "steps": result["steps"], "calls": result["calls"],
           "probe_calls": (len(probe) if probe else 0),
           "confirm_calls": 1,
           "total_calls": result["calls"] + (len(probe) if probe else 0) + 1,
           "capped": result["capped"], "rows": result["rows"],
           "search_score": result["search"], "confirm": verdict}
    name = "combine_v2.json" if a.v2 else "combine.json"
    (job / "trait" / name).write_text(json.dumps(out, indent=1))
    print("WROTE", job / "trait" / name)


if __name__ == "__main__":
    main()
