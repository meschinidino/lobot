"""F1 acceptance: feature coherence median and steering lift (F1).

Per layer: draw 50 live features (seed 7), measure reader coherence
on up-12 and lift at dose 1.0 with the 8 fixed lift prompts; draw 50
random neurons (seed 8) for the single-neuron coherence baseline.
Gate: feature coherence median >= 0.6 AND lift >= 3 on >= half.
Needs real weights: book docs/roadmap/model-lane.md first.

Example:
  uv run python scripts/eval_features.py \\
      --model meta-llama/Llama-3.2-3B-Instruct --layers 12,20
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from model_lobotomy import engine  # noqa: E402
from model_lobotomy import reader  # noqa: E402
from model_lobotomy import sae  # noqa: E402
from model_lobotomy import steer  # noqa: E402

N = 50
FEATURE_SEED = 7
NEURON_SEED = 8
DOSE = 1.0
DIAG_DOSES = (0.5, 2.0)
COHERENCE_GATE = 0.6


def live_features(sae_dir: Path) -> list[int]:
    """Feature indices with max cached act > 0 (fired at least once)."""
    with np.load(sae_dir / sae.FIRES_NAME) as data:
        act = data["act"]
    return [i for i in range(act.shape[0]) if float(act[i].max()) > 0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="F1 acceptance: feature coherence and lift.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", default="12,20")
    parser.add_argument("--n", type=int, default=N)
    args = parser.parse_args(argv)
    prompts_path = ROOT / "store" / "reader" / "lift_prompts.json"
    prompts = json.loads(prompts_path.read_text())
    eng = engine.get_engine()
    print(f"loading {args.model} ...", flush=True)
    eng.load(args.model)
    try:
        spec = eng.spec
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
        preset = None
        from model_lobotomy import app as engine_app
        for mid, pres in engine_app._PRESETS.items():
            if mid == args.model:
                preset = pres
        if preset is None:
            print(f"unknown model {args.model}", file=sys.stderr)
            return 2
        key = steer.calibration_cache_key(
            args.model, eng.revision, eng.tokenizer,
            str(steer.DTYPE), steer.DEVICE)
        with eng.clean():
            medians, _ = steer.calibrate_layer_medians(
                eng.model, eng.tokenizer,
                list(preset["calibration_prompts"]), layers,
                cache_path=ROOT / steer.calibration_path_for(args.model),
                key=key, spec=spec)
        all_pass = True
        for layer in layers:
            sae_dir = sae.sae_dir_for(ROOT / "store", args.model, layer)
            model, cfg = sae.load_sae(sae_dir)
            live = live_features(sae_dir)
            rng = random.Random(FEATURE_SEED)
            feats = rng.sample(live, min(args.n, len(live)))
            coh, lift1, lift05, lift2 = [], [], [], []
            t0 = time.perf_counter()
            for i, feat in enumerate(feats):
                vec = sae.feature_direction(model, feat)
                pushes = reader.pushes_of_direction(
                    eng, vec, k=12, layer=layer)
                coh.append(pushes["coherence"])
                words = sae.top_words(pushes)
                for dose, acc in ((DOSE, lift1), (DIAG_DOSES[0], lift05),
                                  (DIAG_DOSES[1], lift2)):
                    got = reader.steer_lift_vector(
                        eng, layer, vec, dose, words, prompts,
                        {layer: float(medians[layer])})
                    acc.append(got["lift"])
                print(f"L{layer} feat {feat}: coh "
                      f"{coh[-1]:.2f} lift {lift1[-1]:.2f} "
                      f"words {words} ({i + 1}/{len(feats)})", flush=True)
            nrng = random.Random(NEURON_SEED)
            n_coh = []
            for neuron in nrng.sample(range(spec.intermediate),
                                      min(args.n, spec.intermediate)):
                n_coh.append(reader.pushes_toward(
                    eng, layer, neuron, k=12)["coherence"])
            med = statistics.median(coh)
            n_med = statistics.median(n_coh)
            share = sum(v >= 3.0 for v in lift1) / len(lift1)
            gate = med >= COHERENCE_GATE and share >= 0.5
            all_pass = all_pass and gate
            print(f"L{layer}: features={cfg['features']} "
                  f"live={len(live)} dead={cfg['features'] - len(live)}",
                  flush=True)
            print(f"L{layer}: feature coherence median {med:.3f} "
                  f"(neuron baseline {n_med:.3f})", flush=True)
            print(f"L{layer}: lift>=3.0 at dose {DOSE}: "
                  f"{sum(v >= 3.0 for v in lift1)}/{len(lift1)} "
                  f"(diag 0.5: {sum(v >= 3.0 for v in lift05)}, "
                  f"2.0: {sum(v >= 3.0 for v in lift2)})", flush=True)
            print(f"L{layer}: eval {time.perf_counter() - t0:.1f}s "
                  f"gate {'PASS' if gate else 'FAIL'}", flush=True)
        print("OVERALL: PASS" if all_pass else "OVERALL: FAIL", flush=True)
        return 0 if all_pass else 1
    finally:
        eng.unload()


if __name__ == "__main__":
    raise SystemExit(main())
