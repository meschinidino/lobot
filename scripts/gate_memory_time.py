"""Wave-0 performance gate: 8-column filmstrip memory + time.

Measures one 8-dose batched Update on the real weights: wall time,
tokens/s, and driver memory before/after against the 24 GB box
budget. NOTE (O6): chat cache semantics are undecided; this measures
the current implementation (hooks active from prefill) and cannot be
read as a ship decision until O6 is answered.

PASS: the filmstrip completes and driver memory stays under 24 GB.
Writes logs/gate_perf/report.json + report.md. Exit 0/1.

Run:
    uv run python scripts/gate_memory_time.py
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

import engine as engine_mod  # noqa: E402
import steer  # noqa: E402

PROMPT = "Explain how photosynthesis works in one paragraph."
MAX_NEW = 60
SEED = 0
DOSES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
BUDGET_MIB = 24 * 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=steer.LLAMA_MODEL_ID)
    ap.add_argument("--workdir", default=str(ROOT / "logs" / "gate_perf"))
    args = ap.parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    eng = engine_mod.Engine()
    eng.load(args.model)
    print(f"[gate-perf] {args.model} warm (rev {eng.revision[:12]})",
          flush=True)
    try:
        spec = eng.spec
        preset = json.loads(
            (ROOT / "presets" / "llama-refusal.json").read_text())
        key = steer.calibration_cache_key(
            eng.model_id, eng.revision, eng.tokenizer,
            str(steer.DTYPE), steer.DEVICE)
        with eng.clean():
            medians, _ = steer.calibrate_layer_medians(
                eng.model, eng.tokenizer,
                preset["calibration_prompts"], [7],
                cache_path=ROOT / steer.calibration_path_for(
                    eng.model_id), key=key, spec=spec)
        torch.manual_seed(11)
        u = torch.randn(spec.hidden)
        u = tuple((u / u.norm()).tolist())
        sets = [engine_mod.EditSet(vector=(engine_mod.VectorEdit(
            7, u, float(medians[7]), d),)) for d in DOSES]
        driver_before = steer.mps_driver_mib()
        t0 = time.perf_counter()
        out = eng.run_batched(PROMPT, sets, medians=medians,
                              max_new_tokens=MAX_NEW, seed=SEED)
        seconds = time.perf_counter() - t0
        driver_after = steer.mps_driver_mib()
        total_tok = sum(r["new_tokens"] for r in out["rows"])
        passed = (driver_after or 0) < BUDGET_MIB
        result = {"gate": "perf", "model": args.model,
                  "fingerprint": eng.fingerprint(),
                  "columns": len(DOSES), "max_new_tokens": MAX_NEW,
                  "seconds": round(seconds, 1),
                  "total_tokens": total_tok,
                  "tok_per_s": round(total_tok / seconds, 1),
                  "driver_before_mib": driver_before,
                  "driver_after_mib": driver_after,
                  "budget_mib": BUDGET_MIB,
                  "fallback_used": out["fallback_used"],
                  "o6": "pending: measured with hooks active from "
                        "prefill; ship decision needs O6",
                  "pass": passed and not out["fallback_used"]}
    finally:
        eng.unload()
    (workdir / "report.json").write_text(json.dumps(result, indent=2))
    (workdir / "report.md").write_text(
        f"# Perf gate: {'PASS' if result['pass'] else 'FAIL'}\n"
        f"{len(DOSES)} cols, {result['total_tokens']} tok in "
        f"{result['seconds']} s ({result['tok_per_s']} tok/s), driver "
        f"{result['driver_before_mib']} -> "
        f"{result['driver_after_mib']} MiB of {BUDGET_MIB}, fallback "
        f"{result['fallback_used']}\n"
        f"O6 pending: hooks active from prefill (see report.json).\n")
    print(f"[gate-perf] {'PASS' if result['pass'] else 'FAIL'}",
          flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
