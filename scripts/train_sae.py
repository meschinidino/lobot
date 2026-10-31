"""Train one TopK SAE per layer on cached residual acts (F1).

Caches per-token block_out vectors from data/reader_sample.txt,
trains a TopK SAE (expansion 8, top-k 32, 8 epochs), and writes
store/sae/<model-slug>/L{layer}/ (weights, config, fires cache).
Needs real weights: book docs/roadmap/model-lane.md first.

Example:
  uv run python scripts/train_sae.py \\
      --model meta-llama/Llama-3.2-3B-Instruct --layers 12,20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from model_lobotomy import engine  # noqa: E402
from model_lobotomy import sae  # noqa: E402
from model_lobotomy import steer  # noqa: E402
from scripts.build_reader_cache import split_docs  # noqa: E402


def parse_layers(text: str, count: int) -> list[int]:
    layers = [int(x) for x in text.split(",") if x.strip()]
    if not layers:
        raise ValueError("give at least one layer, e.g. --layers 12,20")
    for layer in layers:
        if not 0 <= layer < count:
            raise ValueError(
                f"layer {layer} out of bounds [0,{count})")
    return layers


def peak_rss_mib() -> float | None:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError):
        return None
    # macOS reports bytes, Linux kilobytes.
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train one TopK SAE per layer.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", default="12,20")
    parser.add_argument("--sample", default="data/reader_sample.txt")
    parser.add_argument("--expansion", type=int, default=sae.EXPANSION)
    parser.add_argument("--top-k", type=int, default=sae.TOP_K)
    parser.add_argument("--epochs", type=int, default=sae.EPOCHS)
    parser.add_argument("--batch", type=int, default=sae.BATCH)
    parser.add_argument("--lr", type=float, default=sae.LR)
    parser.add_argument("--seed", type=int, default=sae.TRAIN_SEED)
    parser.add_argument("--target-chars", type=int, default=1000)
    parser.add_argument("--device", default=None,
                        help="default: mps when available, else cpu")
    args = parser.parse_args(argv)
    sample = Path(args.sample)
    if not sample.exists():
        print(f"sample text not found: {sample}", file=sys.stderr)
        return 2
    device = args.device or pick_device()
    docs = split_docs(sample.read_text(encoding="utf-8"),
                      target_chars=args.target_chars)
    eng = engine.get_engine()
    print(f"loading {args.model} ...", flush=True)
    eng.load(args.model)
    try:
        spec = eng.spec
        layers = parse_layers(args.layers, spec.layers)
        features = args.expansion * spec.hidden
        print(f"layers={layers} hidden={spec.hidden} "
              f"features={features} top_k={args.top_k} "
              f"epochs={args.epochs} batch={args.batch} device={device}",
              flush=True)
        for layer in layers:
            t0 = time.perf_counter()
            cached = sae.cache_block_acts(eng, docs, layer)
            n = cached["acts"].shape[0]
            print(f"L{layer}: cached {n} token acts in "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)

            def _log(step, loss):
                if step == 1 or step % 50 == 0:
                    print(f"L{layer}: step {step} mse {loss:.4f}",
                          flush=True)

            t1 = time.perf_counter()
            model, stats = sae.train_sae(
                cached["acts"], hidden=spec.hidden, features=features,
                top_k=args.top_k, epochs=args.epochs, batch=args.batch,
                lr=args.lr, seed=args.seed, device=device, log=_log)
            out = sae.sae_dir_for(ROOT / "store", args.model, layer)
            sae.save_sae(out, model, {
                "model_id": args.model, "revision": eng.revision,
                "layer": layer, "expansion": args.expansion,
                "epochs": args.epochs, "batch": args.batch,
                "lr": args.lr, "sample": str(sample),
                "sample_tokens": len(cached["tokens"]),
                "train_device": device, **stats})
            sae.build_feature_fires(
                model, cached["acts"], cached["tokens"],
                cached["doc_starts"], out, k=8)
            print(f"L{layer}: mse {stats['mse_first']:.4f} -> "
                  f"{stats['mse_last']:.4f} live {stats['live']}/"
                  f"{features} train {time.perf_counter() - t1:.1f}s "
                  f"total {time.perf_counter() - t0:.1f}s -> {out}",
                  flush=True)
            del cached, model
    finally:
        eng.unload()
    mib = peak_rss_mib()
    print(f"peak RSS: {mib:.0f} MiB" if mib else "peak RSS: unknown")
    print(f"torch threads: {torch.get_num_threads()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
