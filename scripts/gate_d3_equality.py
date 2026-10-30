"""Wave-0 gate D3: batched filmstrip == sequential singles, exact.

Loads one model on the Engine and proves Engine.run_batched returns
the same token ids as one Engine.run per row, for a vector-dose
filmstrip and a neuron-dose filmstrip, plus stream parity
(run_batched_stream concatenates to the same texts and ids).

PASS: every case matches exactly with fallback_used False (a
fallback would make the equality vacuous). Writes
logs/gate_d3/report.json + report.md. Exit 0 on pass, 1 on fail.

Run:
    uv run python scripts/gate_d3_equality.py
    uv run python scripts/gate_d3_equality.py --model Qwen/Qwen3.5-4B
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import engine as engine_mod  # noqa: E402
from model_lobotomy import steer  # noqa: E402

PROMPT = "Explain how photosynthesis works in one paragraph."
MAX_NEW = 32
SEED = 0
DOSES = [0.0, 0.5, 1.0]


def _medians(eng, spec, layers):
    preset = json.loads(
        (ROOT / "presets" / "llama-refusal.json").read_text())
    key = steer.calibration_cache_key(
        eng.model_id, eng.revision, eng.tokenizer, str(steer.DTYPE),
        steer.DEVICE)
    with eng.clean():
        medians, _ = steer.calibrate_layer_medians(
            eng.model, eng.tokenizer, preset["calibration_prompts"],
            layers,
            cache_path=ROOT / steer.calibration_path_for(eng.model_id),
            key=key, spec=spec)
    return medians


def _case_vector(eng, spec, medians):
    torch.manual_seed(7)
    u = torch.randn(spec.hidden)
    u = u / u.norm()
    return [engine_mod.EditSet(vector=(engine_mod.VectorEdit(
        7, tuple(u.tolist()), float(medians[7]), d),)) for d in DOSES]


def _case_neuron():
    return [engine_mod.EditSet(neuron=(engine_mod.NeuronEdit(11, 1343, d),))
            for d in [0.0, 1.0, 2.0]]


def _check(eng, name, sets, medians):
    t0 = time.time()
    batched = eng.run_batched(PROMPT, sets, medians=medians,
                              max_new_tokens=MAX_NEW, seed=SEED)
    singles = [eng.run(PROMPT, es, medians=medians,
                       max_new_tokens=MAX_NEW, seed=SEED)
               for es in sets]
    match = all(b["token_ids"] == s["token_ids"]
                for b, s in zip(batched["rows"], singles))
    cols = ["" for _ in sets]
    stream_ids = None
    stream_fallback = None
    for ev in eng.run_batched_stream(
            PROMPT, sets, medians=medians, max_new_tokens=MAX_NEW,
            seed=SEED):
        if ev["type"] == "token":
            cols[ev["col"]] += ev["text"]
        else:
            stream_ids = [r["token_ids"] for r in ev["result"]["rows"]]
            stream_fallback = ev["result"]["fallback_used"]
    stream_match = (
        cols == [r["text"] for r in batched["rows"]]
        and stream_ids == [r["token_ids"] for r in batched["rows"]]
        and stream_fallback == batched["fallback_used"])
    ok = match and stream_match and not batched["fallback_used"]
    return {"case": name, "pass": ok,
            "batched_eq_singles": match,
            "stream_parity": stream_match,
            "fallback_used": batched["fallback_used"],
            "token_ids": [r["token_ids"] for r in batched["rows"]],
            "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=steer.LLAMA_MODEL_ID)
    ap.add_argument("--workdir", default=str(ROOT / "logs" / "gate_d3"))
    args = ap.parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    eng = engine_mod.Engine()
    eng.load(args.model)
    print(f"[gate-d3] {args.model} warm (rev {eng.revision[:12]})",
          flush=True)
    try:
        spec = eng.spec
        medians = _medians(eng, spec, [7, 11])
        cases = [_check(eng, "vector-doses",
                        _case_vector(eng, spec, medians), medians),
                 _check(eng, "neuron-doses", _case_neuron(), medians)]
        out = {"gate": "D3", "model": args.model,
               "fingerprint": eng.fingerprint(),
               "prompt": PROMPT, "max_new_tokens": MAX_NEW,
               "cases": cases,
               "pass": all(c["pass"] for c in cases)}
    finally:
        eng.unload()
    (workdir / "report.json").write_text(json.dumps(out, indent=2))
    lines = [f"# Gate D3: {'PASS' if out['pass'] else 'FAIL'}",
             f"model {args.model} rev "
             f"{out['fingerprint'].get('revision', '?')[:12]}"]
    for c in cases:
        lines.append(f"- {c['case']}: batched==singles "
                     f"{c['batched_eq_singles']}, stream "
                     f"{c['stream_parity']}, fallback "
                     f"{c['fallback_used']} ({c['seconds']} s)")
    (workdir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[gate-d3] {'PASS' if out['pass'] else 'FAIL'}", flush=True)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
