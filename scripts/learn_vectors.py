"""ML v3 live run: learn steering vectors for one family.

Usage: uv run python scripts/learn_vectors.py <family> <job-dir>
[--model meta-llama/Llama-3.2-3B-Instruct]

Trains one vector per probe-top-8 layer (model frozen, drift cost
on), scores dose scales on the tuning split, confirms the best on
the brand-new questions in fresh_questions_v3.json, and writes
learned.json plus learned_vectors.pt next to the family report.
Gates: docs/roadmap/specs/ML-multi-layer-search.md (v3 amendment).
Take the model lane before running.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import torch
from model_lobotomy import app as app_mod
from model_lobotomy import directions, engine as engine_mod
from model_lobotomy import learned, pareto, steer, trait_eval
from model_lobotomy import layer_search as combine

LAYERS = [11, 9, 12, 10, 13, 14, 15, 16]
SCALES = [0.5, 0.75, 1.0, 1.25]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("job")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--vectors", default=None,
                    help="skip training; load these learned vectors")
    ap.add_argument("--scale", type=float, default=None,
                    help="confirm only this dose scale (no search)")
    ap.add_argument("--fresh", default="fresh_questions_v3.json",
                    help="confirm question file in the trait dir")
    ap.add_argument("--shots", default=None,
                    help="few-shot examples {q, a} for the positive "
                         "side, as a JSON file in the trait dir")
    ap.add_argument("--drift-w", type=float, default=None,
                    help="drift cost weight (default: learned.DRIFT_W); "
                         "higher keeps text cleaner at some strength")
    a = ap.parse_args()
    job = ROOT / a.job if not Path(a.job).is_absolute() else Path(a.job)
    report = json.loads((job / "trait" / "report.json").read_text())
    cfg = trait_eval.family_cfg(a.family)
    tuning = report["splits"]["tuning"]
    fresh = json.loads(
        (job / "trait" / a.fresh).read_text())["questions"]

    app_mod._active_model_id = a.model
    preset, spec = app_mod.active_preset(), app_mod.active_spec()
    eng = engine_mod.Engine()
    eng.load(a.model)
    judge = trait_eval.LocalJudge(eng.model, eng.tokenizer, spec)
    port = directions.get_direction()
    A, B = port.derive(eng, spec, cfg, report["splits"]["extraction"])
    seps = port.seps(A, B, list(range(spec.layers)))
    key = steer.calibration_cache_key(a.model, eng.revision,
                                      eng.tokenizer, str(steer.DTYPE),
                                      steer.DEVICE)
    medians, _ = steer.calibrate_layer_medians(
        eng.model, eng.tokenizer, preset["calibration_prompts"], LAYERS,
        cache_path=ROOT / steer.calibration_path_for(a.model),
        key=key, spec=spec)

    if a.scale is not None and not (a.scale == a.scale and abs(a.scale) != float("inf")):
        ap.error("--scale must be a finite number")
    if a.drift_w is not None and not (
            a.drift_w == a.drift_w and abs(a.drift_w) != float("inf")
            and a.drift_w >= 0):
        ap.error("--drift-w must be a finite number at or above 0")
    if a.vectors:
        # vectors_only: a .pt file must never run pickle code.
        raw = torch.load(a.vectors, weights_only=True)
        try:
            vecs = learned.validate_vectors(raw, LAYERS)
        except ValueError as e:
            raise SystemExit(f"bad learned vectors in {a.vectors}: {e}")
        hist = []
        print("LOADED VECTORS", a.vectors, flush=True)
        _score_and_write(eng, judge, cfg, medians, vecs, hist,
                         tuning, fresh, job, a)
        return
    shots = []
    if a.shots:
        shots = json.loads((job / "trait" / a.shots).read_text())["examples"]
        print("FEW-SHOT POSITIVE SIDE:", len(shots), "examples", flush=True)
    # Contrast answers from the plain model, one pair per question.
    print("GENERATING CONTRAST ANSWERS", flush=True)
    pairs_text = []
    for q in tuning:
        row = {}
        for side, ins in (("pos", cfg["pos"]), ("neg", cfg["neg"])):
            prompt = (learned.fewshot_prompt(ins, shots, q)
                      if side == "pos" else f"{ins}\n\n{q}")
            r = eng.run_batched(prompt,
                                [engine_mod.EditSet()],
                                medians=medians, max_new_tokens=60,
                                seed=app_mod.SEED)
            row[side] = r["rows"][0]["text"]
        pairs_text.append((q, row["pos"], row["neg"]))
    tok = eng.tokenizer

    def encode(q, ans):
        p_ids = tok(steer.wrap_prompt_notk(tok, q, spec),
                    add_special_tokens=False)["input_ids"]
        a_ids = tok(ans, add_special_tokens=False)["input_ids"]
        return {"input_ids": torch.tensor(p_ids + a_ids),
                "answer_start": len(p_ids)}

    seqs, drift = [], []
    for i, (q, pos, neg) in enumerate(pairs_text):
        s = encode(q, pos); s["side"] = "pos"; s["pair"] = i
        seqs.append(s)
        s = encode(q, neg); s["side"] = "neg"; s["pair"] = i
        seqs.append(s)
    for q in tuning[:5]:
        drift.append({"input_ids": torch.tensor(
            tok(steer.wrap_prompt_notk(tok, q, spec),
                add_special_tokens=False)["input_ids"])})

    for p in eng.model.parameters():
        p.requires_grad_(False)
    mods = {L: steer.resolve_block(eng.model, L, spec) for L in LAYERS}
    init = {L: (torch.as_tensor(seps[L]["unit"], dtype=torch.float32)
                * float(medians[L]) * 0.25) for L in LAYERS}
    print("TRAINING", flush=True)
    vecs, hist = learned.train_vectors(
        eng.model, mods, seqs, drift, init,
        drift_w=a.drift_w if a.drift_w is not None else learned.DRIFT_W,
        log=lambda r: print(" ", r, flush=True))
    torch.save(vecs, job / "trait" / "learned_vectors.pt")
    (job / "trait" / "learned_history.json").write_text(
        json.dumps(hist, indent=1))
    _score_and_write(eng, judge, cfg, medians, vecs, hist, tuning,
                     fresh, job, a)


def _score_and_write(eng, judge, cfg, medians, vecs, hist, tuning,
                     fresh, job, a):
    base_cache = {}

    def score_factory(prompts):
        if id(prompts) not in base_cache:
            rows = []
            for p in prompts:
                r = eng.run_batched(p, [engine_mod.EditSet()],
                                    medians=medians, max_new_tokens=100,
                                    seed=app_mod.SEED)
                t = r["rows"][0]["text"]
                rows.append(judge.score(p, t, cfg["judge"]))
            base_cache[id(prompts)] = rows
        jb = base_cache[id(prompts)]

        def score(scale):
            edit = engine_mod.EditSet(vector=tuple(
                engine_mod.VectorEdit(
                    L, tuple((vecs[L] / vecs[L].norm()).tolist()),
                    float(vecs[L].norm()) * scale, 1.0)
                for L in LAYERS))
            gains, flus = [], []
            for p, b in zip(prompts, jb):
                r = eng.run_batched(p, [edit], medians=medians,
                                    max_new_tokens=100,
                                    seed=app_mod.SEED)
                t = r["rows"][0]["text"]
                gains.append(judge.score(p, t, cfg["judge"]) - b)
                flus.append(judge.score(p, t, pareto.FLUENCY_LABEL))
            return {"gain": round(sum(gains) / len(gains), 3),
                    "fluency_mean": round(sum(flus) / len(flus), 3),
                    "fluency_min": round(min(flus), 3),
                    "fluency_p10": round(trait_eval._p10(flus), 3)}
        return score

    search_score = score_factory(tuning[:10])
    if a.scale is not None:
        # Fixed-scale confirm (v3b shape): no search, no picking.
        fs = score_factory(fresh)(a.scale)
        passed = fs["gain"] >= trait_eval.E.MIN_GAIN and combine.fluency_ok(fs)
        print("FIXED SCALE", a.scale, "CONFIRM", fs,
              "PASS" if passed else "FAIL", flush=True)
        _write_patch(job, a, vecs, {"passed": passed, "score": fs,
                                    "facets": trait_eval.label_facets(cfg["judge"]),
                                    "fresh_prompts": len(fresh),
                                    "set": a.fresh, "scale": a.scale})
        _write(job, a, hist, vecs, {}, a.scale,
               {"passed": passed, "score": fs, "scale": a.scale,
                "facets": trait_eval.label_facets(cfg["judge"]),
                "fresh_prompts": len(fresh), "set": a.fresh})
        return
    results = {}
    for s in SCALES:
        results[str(s)] = search_score(s)
        print("SCALE", s, results[str(s)], flush=True)
    ok = [s for s in SCALES
          if combine.fluency_ok(results[str(s)])]
    # If no scale is fluent, the best-gain scale still confirms and
    # fails there; the confirm, not the search, decides.
    best = max(ok or SCALES, key=lambda s: results[str(s)]["gain"])
    fresh_score = score_factory(fresh)
    fs = fresh_score(best)
    passed = fs["gain"] >= trait_eval.E.MIN_GAIN and combine.fluency_ok(fs)
    print("BEST SCALE", best, "CONFIRM", fs, "PASS" if passed else "FAIL",
          flush=True)
    _write_patch(job, a, vecs, {"passed": passed, "score": fs,
                                "facets": trait_eval.label_facets(cfg["judge"]),
                                "fresh_prompts": len(fresh),
                                "set": a.fresh, "scale": best})
    _write(job, a, hist, vecs, results, best,
           {"passed": passed, "score": fs, "scale": best,
            "facets": trait_eval.label_facets(cfg["judge"]),
            "fresh_prompts": len(fresh), "set": a.fresh})


def _known_guard(job, a):
    # A resumed run reuses the same vectors, so the recorded guard
    # verdict still describes this mix. A fresh training run makes
    # a new mix: the old verdict does not carry over.
    if a.vectors:
        p = job / "trait" / "learned.json"
        if p.is_file():
            g = json.loads(p.read_text()).get("guard")
            if g:
                return g
    return {"status": "not_run_for_this_mix"}


def _write_patch(job, a, vecs, confirm):
    """Save the passing mix as a multi-vector patch for the app."""
    if not confirm.get("passed"):
        return
    from model_lobotomy import patch as patch_mod
    report = json.loads((job / "trait" / "report.json").read_text())
    patch_mod.save_multi_patch(
        job / "trait" / "learned.lobot", vectors=vecs,
        dose=float(confirm.get("scale") or 0.625), trait=a.family,
        recipe={"kind": "learned", "layers": LAYERS,
                "method": "scripts/learn_vectors.py"},
        report_card={"gain": confirm["score"]["gain"],
                     "fluency_mean": confirm["score"]["fluency_mean"],
                     "fluency_p10": confirm["score"]["fluency_p10"],
                     "fresh_prompts": confirm["fresh_prompts"],
                     "set": confirm.get("set")},
        guard=_known_guard(job, a),
        fingerprint={"model_id": a.model,
                     "revision": report.get("revision")})
    print("WROTE", job / "trait" / "learned.lobot")


def _write(job, a, hist, vecs, results, best, confirm):
    prev = {}
    p = job / "trait" / "learned.json"
    if p.is_file():
        prev = json.loads(p.read_text())
    out = {"family": a.family, "model": a.model, "layers": LAYERS,
           # A resume run has no history of its own; keep the
           # training run's, never overwrite it with empty.
           "history_tail": hist[-5:] or prev.get("history_tail", []),
           "steps": len(hist) or prev.get("steps", 0),
           "norms": {str(L): round(float(vecs[L].norm()), 2)
                     for L in LAYERS},
           # Keep the guard verdict for this mix. A resume reuses
           # it; a fresh mix records that no guard ran yet. Never
           # erase it: the UI badge reads this field.
           "guard": prev.get("guard") if a.vectors else
           {"status": "not_run_for_this_mix"},
           "search_scores": results or prev.get("search_scores", {}),
           "best_scale": best,
           "confirm": confirm,
           "confirm_v3b": prev.get("confirm_v3b"),
           "exploratory_v3": prev.get("exploratory_v3")}
    p.write_text(json.dumps(out, indent=1))
    print("WROTE", p)


if __name__ == "__main__":
    main()
