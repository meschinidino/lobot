"""Parity + speed + guard: torch Engine vs LlamaEngine (D5 parts 2-5).

Sequential phases (one backend warm at a time): torch runs first,
unloads, then llama runs the same prompts x editsets. Compares:

- 32 greedy token ids per prompt x {baseline, vector, neuron,
  vector+neuron}: exact match, else first split at index >= 16
  with torch logit gap < 0.5 (near-tie).
- Prefill per prompt x editset: full-vocab cosine >= 0.999,
  same top-1, refusal-starter |d| <= 0.01 (F16 strict bars).
- Guard().check verdicts on 3 neuron doses (0/1/2): identical.
- Speed: median of 5 baseline runs (TTFT, tok/s) + cvec build ms.

--quant q8_0 switches to the loose bars (baseline prefill top-1
rate >= 90%, refusal |d| <= 0.03, identical verdicts, no token
bar). --quant q4_k_m is report-only.

Writes <workdir>/report.json + report.md. Exit 0/1.

Run:
    uv run python scripts/parity_llama.py --model meta-llama/Llama-3.2-3B-Instruct
    uv run python scripts/parity_llama.py --model meta-llama/Llama-3.2-3B-Instruct --quant q8_0
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import engine as engine_mod  # noqa: E402
from model_lobotomy import guard as guard_mod  # noqa: E402
from model_lobotomy import llama_backend as lb  # noqa: E402
from model_lobotomy import patch as patch_mod  # noqa: E402
from model_lobotomy import steer  # noqa: E402

PROMPTS = [
    "Explain how photosynthesis works in one paragraph.",
    "Who are you, really?",
    "Write a haiku about the sea.",
]
GUARD_DOSES = [0.0, 1.0, 2.0]
SPEED_REPS = 5


def _knob_default(model_id: str) -> str:
    if model_id == steer.LLAMA_MODEL_ID:
        return "jobs/pirate-0-20261030-004543-000/trait/knob.lobot"
    return "jobs/cheerful-0-20261030-134601-000/trait/knob.lobot"


def _preset_row(model_id: str):
    for p in sorted((ROOT / "presets").glob("*.json")):
        d = json.loads(p.read_text())
        if d.get("model_id") == model_id:
            r = d["rows"][0]
            return (r["layer"], r["neuron"],
                    list(d["calibration_prompts"]))
    raise RuntimeError(f"no preset for {model_id}")


def _torch_gaps(eng, prompt_ids, gen_ids, edits, medians):
    """Per-position (top1, top2, gap) under edits, raw torch.

    Replicates greedy generate step by step inside apply_edits:
    one forward per generated token. Returns three lists.
    """
    spec = eng.spec
    top1, top2, gaps = [], [], []
    with engine_mod.apply_edits(eng.model, eng.tokenizer, spec,
                                edits, medians):
        with torch.no_grad():
            dev = str(next(eng.model.parameters()).device)
            for i in range(len(gen_ids)):
                ids = torch.tensor(
                    [prompt_ids + gen_ids[:i]],
                    dtype=torch.long).to(dev)
                logits = eng.model(ids).logits[0, -1].float()
                vals, idx = torch.topk(logits, 2)
                top1.append(int(idx[0]))
                top2.append(int(idx[1]))
                gaps.append(float(vals[0] - vals[1]))
    return top1, top2, gaps


def _refusal_mass(logits_np, refusal_ids) -> float:
    t = torch.from_numpy(np.asarray(logits_np, dtype=np.float32))
    probs = torch.softmax(t, dim=-1)
    return float(sum(probs[i].item() for i in refusal_ids
                     if i < probs.numel()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=steer.LLAMA_MODEL_ID)
    ap.add_argument("--quant", default="f16",
                    choices=["f16", "q8_0", "q4_k_m"])
    ap.add_argument("--knob", default=None)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()
    strict = args.quant == "f16"
    report_only = args.quant == "q4_k_m"
    workdir = Path(args.workdir or
                   (ROOT / "logs" / "parity_llama" / args.quant))
    workdir.mkdir(parents=True, exist_ok=True)
    out: dict = {"model": args.model, "quant": args.quant,
                 "strict": strict, "pass": False}

    def fail(reason):
        out["fail_reason"] = reason
        (workdir / "report.json").write_text(
            json.dumps(out, indent=2))
        (workdir / "report.md").write_text(
            f"# Parity {args.quant}: FAIL\n{reason}\n")
        print(f"[parity] FAIL: {reason}", flush=True)
        return 1

    knob_path = ROOT / (args.knob or _knob_default(args.model))
    if not knob_path.is_file():
        return fail(f"knob missing: {knob_path}")
    try:
        loaded = patch_mod.load_patch(knob_path)
    except patch_mod.PatchRejected as e:
        return fail(f"knob rejected: {e}")
    meta = loaded["meta"]
    klayer, kdose = int(meta["layer"]), float(meta["dose"])
    kscale = float(meta["scale"])
    kunit = tuple(loaded["direction"].detach().cpu().tolist())
    nlayer, nneuron, calib_prompts = _preset_row(args.model)
    print(f"[parity] knob L{klayer}@{kdose} scale {kscale:.2f}; "
          f"neuron L{nlayer}N{nneuron}", flush=True)

    vec = engine_mod.VectorEdit(klayer, kunit, kscale, kdose)
    neu = engine_mod.NeuronEdit(nlayer, nneuron, 1.0)
    sets = {"baseline": engine_mod.EMPTY_EDITS,
            "vector": engine_mod.EditSet(vector=(vec,)),
            "neuron": engine_mod.EditSet(neuron=(neu,)),
            "both": engine_mod.EditSet(neuron=(neu,),
                                       vector=(vec,))}

    # -- torch phase --
    teng = engine_mod.Engine()
    teng.load(args.model)
    print(f"[parity] torch warm rev {teng.revision[:12]}",
          flush=True)
    try:
        spec = teng.spec
        key = steer.calibration_cache_key(
            args.model, teng.revision, teng.tokenizer,
            str(steer.DTYPE), steer.DEVICE)
        medians, cal_info = steer.calibrate_layer_medians(
            teng.model, teng.tokenizer, calib_prompts,
            sorted({nlayer, klayer}),
            cache_path=ROOT / steer.calibration_path_for(
                args.model), key=key, spec=spec)
        print(f"[parity] medians cache_hit="
              f"{cal_info['cache_hit']}", flush=True)
        refusal_ids = guard_mod.refusal_id_set(teng.tokenizer)
        torch_runs, torch_pres = {}, {}
        for pi, prompt in enumerate(PROMPTS):
            wrapped = steer.wrap_prompt_notk(
                teng.tokenizer, prompt, spec)
            prompt_ids = teng.tokenizer(
                wrapped, return_tensors="pt")["input_ids"][0]
            prompt_ids = [int(x) for x in prompt_ids.tolist()]
            for name, edits in sets.items():
                gen = teng.run(prompt, edits, medians=medians,
                               max_new_tokens=args.max_new,
                               seed=0)
                g1, g2, gaps = _torch_gaps(
                    teng, prompt_ids, gen["token_ids"], edits,
                    medians)
                torch_runs[(pi, name)] = {
                    "ids": gen["token_ids"],
                    "top1": g1, "top2": g2, "gaps": gaps,
                    "prefill_s": gen["prefill_s"],
                    "tok_per_s": gen["tok_per_s"],
                    "seconds": gen["seconds"]}
                logits = teng.prefill(
                    prompt, edits, medians=medians).detach().cpu().numpy()
                torch_pres[(pi, name)] = {
                    "logits": np.asarray(logits,
                                         dtype=np.float32),
                    "top1": int(np.argmax(logits)),
                    "refusal": _refusal_mass(logits,
                                             refusal_ids)}
                print(f"[parity] torch p{pi} {name}: "
                      f"{gen['new_tokens']} tok, "
                      f"{gen['tok_per_s']} tok/s", flush=True)
        torch_guard = []
        for dose in GUARD_DOSES:
            es = engine_mod.EditSet(neuron=(
                engine_mod.NeuronEdit(nlayer, nneuron, dose),))
            gv = guard_mod.Guard().check(
                teng, es, medians=medians)
            torch_guard.append(
                {"dose": dose, "blocked": gv.blocked,
                 "min_ratio": gv.min_ratio,
                 "would_block": gv.would_block})
        torch_speed = []
        for _ in range(SPEED_REPS):
            gen = teng.run(PROMPTS[0], engine_mod.EMPTY_EDITS,
                           medians=medians,
                           max_new_tokens=args.max_new, seed=0)
            torch_speed.append(
                {"wall": gen["seconds"],
                 "prefill_s": gen["prefill_s"],
                 "tok_per_s": gen["tok_per_s"]})
        out["torch_fingerprint"] = teng.fingerprint()
    finally:
        teng.unload()

    # -- llama phase --
    os.environ["LOBOT_LLAMA_QUANT"] = args.quant
    leng = lb.LlamaEngine()
    leng.load(args.model)
    print(f"[parity] llama warm {args.quant}", flush=True)
    try:
        llama_runs, llama_pres = {}, {}
        for pi, prompt in enumerate(PROMPTS):
            for name, edits in sets.items():
                gen = leng.run(prompt, edits, medians=medians,
                               max_new_tokens=args.max_new,
                               seed=0)
                llama_runs[(pi, name)] = {
                    "ids": gen["token_ids"],
                    "prefill_s": gen["prefill_s"],
                    "tok_per_s": gen["tok_per_s"],
                    "seconds": gen["seconds"]}
                logits = leng.prefill(
                    prompt, edits, medians=medians).detach().cpu().numpy()
                llama_pres[(pi, name)] = {
                    "logits": np.asarray(logits,
                                         dtype=np.float32),
                    "top1": int(np.argmax(logits)),
                    "refusal": _refusal_mass(logits,
                                             refusal_ids)}
                print(f"[parity] llama p{pi} {name}: "
                      f"{gen['new_tokens']} tok, "
                      f"{gen['tok_per_s']} tok/s", flush=True)
        llama_guard = []
        for dose in GUARD_DOSES:
            es = engine_mod.EditSet(neuron=(
                engine_mod.NeuronEdit(nlayer, nneuron, dose),))
            gv = guard_mod.Guard().check(
                leng, es, medians=medians)
            llama_guard.append(
                {"dose": dose, "blocked": gv.blocked,
                 "min_ratio": gv.min_ratio,
                 "would_block": gv.would_block})
        llama_speed = []
        for _ in range(SPEED_REPS):
            gen = leng.run(PROMPTS[0], engine_mod.EMPTY_EDITS,
                           medians=medians,
                           max_new_tokens=args.max_new, seed=0)
            llama_speed.append(
                {"wall": gen["seconds"],
                 "prefill_s": gen["prefill_s"],
                 "tok_per_s": gen["tok_per_s"]})
        t0 = time.perf_counter()
        for _ in range(20):
            leng._cvec_for(sets["both"], medians)
        cvec_ms = (time.perf_counter() - t0) / 20 * 1000
        out["llama_fingerprint"] = leng.fingerprint()
    finally:
        leng.unload()

    # -- compare --
    gen_rows, pre_rows = [], []
    for key in torch_runs:
        pi, name = key
        t_ids = torch_runs[key]["ids"]
        l_ids = llama_runs[key]["ids"]
        n = min(len(t_ids), len(l_ids), args.max_new)
        div = next((i for i in range(n)
                    if t_ids[i] != l_ids[i]), None)
        if div is None and len(t_ids) != len(l_ids):
            div = n
        gaps = torch_runs[key]["gaps"]
        if div is None:
            verdict = "exact"
        elif div >= 16 and gaps[div] < 0.5:
            verdict = f"near-tie@{div}(gap {gaps[div]:.3f})"
        else:
            verdict = (f"SPLIT@{div}"
                       + (f"(gap {gaps[div]:.3f})"
                          if div < len(gaps) else "(len)"))
        gen_rows.append({"prompt": pi, "edits": name,
                         "verdict": verdict,
                         "torch_n": len(t_ids),
                         "llama_n": len(l_ids)})
        tl = torch_pres[key]
        ll = llama_pres[key]
        cos = float(
            np.dot(tl["logits"], ll["logits"])
            / (np.linalg.norm(tl["logits"])
               * np.linalg.norm(ll["logits"])))
        pre_rows.append(
            {"prompt": pi, "edits": name,
             "cosine": round(cos, 6),
             "top1_match": tl["top1"] == ll["top1"],
             "refusal_d": round(abs(tl["refusal"]
                                    - ll["refusal"]), 4)})
    guard_rows = [
        {"dose": t["dose"],
         "torch_blocked": t["blocked"],
         "llama_blocked": l["blocked"],
         "torch_ratio": round(t["min_ratio"], 4),
         "llama_ratio": round(l["min_ratio"], 4)}
        for t, l in zip(torch_guard, llama_guard)]

    def med(xs):
        return round(statistics.median(xs), 2)

    speed = {
        "torch_wall": med([s["wall"] for s in torch_speed]),
        "torch_ttft": med([s["prefill_s"]
                           for s in torch_speed]),
        "torch_tps": med([s["tok_per_s"]
                          for s in torch_speed]),
        "llama_wall": med([s["wall"] for s in llama_speed]),
        "llama_ttft": med([s["prefill_s"]
                           for s in llama_speed]),
        "llama_tps": med([s["tok_per_s"]
                          for s in llama_speed]),
        "cvec_build_ms": round(cvec_ms, 2)}
    out.update({"gen": gen_rows, "prefill": pre_rows,
                "guard": guard_rows, "speed": speed})

    # -- bars --
    if strict and not report_only:
        gen_ok = all(not r["verdict"].startswith("SPLIT")
                     for r in gen_rows)
        pre_ok = all(r["cosine"] >= 0.999
                     and r["top1_match"]
                     and r["refusal_d"] <= 0.01
                     for r in pre_rows)
        guard_ok = all(r["torch_blocked"] == r["llama_blocked"]
                       for r in guard_rows)
        out.update({"gen_ok": gen_ok, "prefill_ok": pre_ok,
                    "guard_ok": guard_ok,
                    "pass": bool(gen_ok and pre_ok
                                 and guard_ok)})
    else:
        base = [r for r in pre_rows if r["edits"] == "baseline"]
        top1_rate = (sum(r["top1_match"] for r in base)
                     / max(len(base), 1))
        max_rd = max([r["refusal_d"] for r in base] or [0.0])
        guard_ok = all(r["torch_blocked"] == r["llama_blocked"]
                       for r in guard_rows)
        meets = bool(top1_rate >= 0.90 and max_rd <= 0.03
                     and guard_ok)
        out.update({"top1_rate": round(top1_rate, 3),
                    "max_refusal_d": round(max_rd, 4),
                    "guard_ok": guard_ok,
                    "meets_q8_bar": meets,
                    # report-only quants always exit 0
                    "pass": meets if not report_only else True})
    (workdir / "report.json").write_text(json.dumps(out, indent=2))
    lines = [f"# Parity {args.quant}: "
             f"{'PASS' if out['pass'] else 'FAIL'}",
             f"model {args.model} strict={strict}"]
    for r in gen_rows:
        lines.append(f"gen p{r['prompt']} {r['edits']}: "
                     f"{r['verdict']}")
    for r in pre_rows:
        lines.append(f"pre p{r['prompt']} {r['edits']}: cos "
                     f"{r['cosine']} top1={r['top1_match']} "
                     f"rd={r['refusal_d']}")
    for r in guard_rows:
        lines.append(f"guard @{r['dose']}: torch "
                     f"{r['torch_blocked']}({r['torch_ratio']}) "
                     f"llama {r['llama_blocked']}"
                     f"({r['llama_ratio']})")
    lines.append(f"speed {speed}")
    (workdir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[parity] {'PASS' if out['pass'] else 'FAIL'}",
          flush=True)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
