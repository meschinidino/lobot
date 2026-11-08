"""Measure knob traits on fresh prompts (not in the prompt bank).

For each trait family: plain vs knob-dose answers on 10 fresh prompts,
scored with the local judge (trait gain), the fluency judge, and a
repetition (collapse) check. Writes jobs/fresh-measure.json and prints
one line per trait.

Run: uv run python scripts/measure_fresh.py [family ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_lobotomy import app as app_mod  # noqa: E402
from model_lobotomy import engine as engine_mod  # noqa: E402
from model_lobotomy import jobs_store, pareto, prompt_bank, steer  # noqa: E402
from model_lobotomy import trait_eval  # noqa: E402

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
FAMILIES = ("talks-like-an-old-west-cowboy", "pirate", "shakespeare",
            "talks-like-an-excited-sports-commentator")
PROMPTS = (
    "What's the most important thing someone should learn?",
    "How should I spend a free Saturday?",
    "What makes a good friend?",
    "Describe a perfect cup of coffee.",
    "What should I do if my bike tire goes flat?",
    "Tell me about a small town you know.",
    "How do I learn to cook rice properly?",
    "What is the best way to save money each month?",
    "Describe the sea at sunrise.",
    "What makes a house feel like home?",
)
MAX_NEW = 120


def latest_knob(family: str) -> dict:
    jobs = []
    for path in (ROOT / "jobs").glob("*/job.json"):
        job = json.loads(path.read_text())
        if (job.get("trait") == family and job.get("status") == "done"
                and job.get("frozen")
                and job.get("model_id") == MODEL_ID):
            jobs.append(job)
    jobs.sort(key=lambda j: j.get("updated", 0), reverse=True)
    job = jobs[0]
    from safetensors.torch import load_file
    direction = load_file(
        str(jobs_store.JobPaths(job["id"]).direction))["direction"]
    return {"layer": job["frozen"]["layer"],
            "dose": job["frozen"]["dose"],
            "direction": direction.tolist(), "job_id": job["id"]}


def repetition(text: str) -> float:
    words = text.lower().split()
    if len(words) < 10:
        return 0.0
    from collections import Counter
    return Counter(words).most_common(1)[0][1] / len(words)


def main() -> int:
    # Args: family or family@dose (dose overrides the frozen knob).
    # --prompt "text" replaces the fresh prompt set (repeatable).
    global PROMPTS
    overrides = {}
    fams = []
    argv = list(sys.argv[1:])
    custom = []
    while "--prompt" in argv:
        i = argv.index("--prompt")
        custom.append(argv[i + 1])
        del argv[i:i + 2]
    out_name = None
    if "--out" in argv:
        i = argv.index("--out")
        out_name = argv[i + 1]
        del argv[i:i + 2]
    if "--prompts-file" in argv:
        i = argv.index("--prompts-file")
        custom = json.loads(
            Path(argv[i + 1]).read_text())["questions"]
        del argv[i:i + 2]
    if custom:
        PROMPTS = tuple(custom)
    for arg in argv:
        fam, at, dose = arg.partition("@")
        fams.append(fam)
        if at:
            overrides[fam] = float(dose)
    families = tuple(fams) or FAMILIES
    bank = set(prompt_bank.PROMPT_BANK)
    assert not any(p in bank for p in PROMPTS), "prompt in bank"
    app_mod._active_model_id = MODEL_ID
    preset = app_mod.active_preset()
    spec = app_mod.active_spec()
    eng = engine_mod.Engine()
    eng.load(MODEL_ID)
    judge = trait_eval.LocalJudge(eng.model, eng.tokenizer, spec)
    out = {"model": MODEL_ID, "prompts": list(PROMPTS), "traits": {}}
    try:
        layers = [latest_knob(f)["layer"] for f in families]
        key = steer.calibration_cache_key(
            MODEL_ID, eng.revision, eng.tokenizer, str(steer.DTYPE),
            steer.DEVICE)
        medians, _ = steer.calibrate_layer_medians(
            eng.model, eng.tokenizer, preset["calibration_prompts"],
            layers, cache_path=ROOT / steer.calibration_path_for(MODEL_ID),
            key=key, spec=spec)
        for family in families:
            knob = latest_knob(family)
            if family in overrides:
                knob = {**knob, "dose": overrides[family]}
            cfg = trait_eval.family_cfg(family)
            vec = engine_mod.VectorEdit(
                knob["layer"], tuple(knob["direction"]),
                float(medians[knob["layer"]]), knob["dose"])
            plain_edit = engine_mod.EditSet(vector=(
                engine_mod.VectorEdit(vec.layer, vec.direction,
                                      vec.scale, 0.0),))
            edit = engine_mod.EditSet(vector=(vec,))
            rows = []
            for prompt in PROMPTS:
                res = eng.run_batched(
                    prompt, [plain_edit, edit],
                    medians=medians, max_new_tokens=MAX_NEW,
                    seed=app_mod.SEED)
                plain, steered = res["rows"][0], res["rows"][1]
                gain = (judge.score(prompt, steered["text"], cfg["judge"])
                        - judge.score(prompt, plain["text"], cfg["judge"]))
                flu = judge.score(prompt, steered["text"],
                                  pareto.FLUENCY_LABEL)
                rep = repetition(steered["text"])
                rows.append({"prompt": prompt, "gain": round(gain, 3),
                             "fluency": round(flu, 3),
                             "repetition": round(rep, 3),
                             "collapsed": rep > 0.5,
                             "plain": plain["text"],
                             "steered": steered["text"]})
            n = len(rows)
            from model_lobotomy import layer_search
            agg = {
                "job_id": knob["job_id"], "layer": knob["layer"],
                "dose": knob["dose"],
                "fluency_p10": round(trait_eval._p10(
                    [r["fluency"] for r in rows]), 3),
                "passed": bool(
                    sum(r["gain"] for r in rows) / n
                    >= trait_eval.E.MIN_GAIN
                    and layer_search.fluency_ok({
                        "fluency_mean": sum(
                            r["fluency"] for r in rows) / n,
                        "fluency_p10": trait_eval._p10(
                            [r["fluency"] for r in rows])})),
                "mean_gain": round(sum(r["gain"] for r in rows) / n, 3),
                "mean_fluency": round(
                    sum(r["fluency"] for r in rows) / n, 3),
                "collapsed": sum(r["collapsed"] for r in rows),
                "weak": sum(r["gain"] < 0.1 for r in rows),
                "rows": rows,
            }
            out["traits"][family] = agg
            print(f"{family}: dose {knob['dose']} gain "
                  f"{agg['mean_gain']} fluency {agg['mean_fluency']} "
                  f"collapsed {agg['collapsed']}/{n} weak "
                  f"{agg['weak']}/{n}", flush=True)
    finally:
        eng.unload()
    path = ROOT / "jobs" / (
        out_name or ("fresh-measure-custom.json" if custom
                     else "fresh-measure.json"))
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
