"""Build the fires_on cache for one model (R1).

Reads data/reader_sample.txt, splits it into short documents, and
writes store/reader/<model-slug>/ with reader.build_fires_on_cache.
Needs real weights: book docs/roadmap/model-lane.md first.

Example:
  uv run python scripts/build_reader_cache.py \\
      --model meta-llama/Llama-3.2-3B-Instruct
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import engine  # noqa: E402
from model_lobotomy import reader  # noqa: E402


def slug_for(model_id: str) -> str:
    return model_id.lower().replace("/", "-").replace("_", "-")


def parse_layers(text: str, count: int) -> list[int] | None:
    if text == "all":
        return None
    if "-" in text:
        lo, hi = text.split("-", 1)
        layers = list(range(int(lo), int(hi) + 1))
    else:
        layers = [int(x) for x in text.split(",") if x.strip()]
    for layer in layers:
        if not 0 <= layer < count:
            raise ValueError(
                f"layer {layer} out of bounds [0,{count})")
    return layers


def split_docs(text: str, target_chars: int = 1000) -> list[str]:
    """Group paragraphs into documents of about target_chars.

    Long paragraphs split at word gaps first, so no document grows
    past about twice the target (the cache build holds one document
    across all layers).
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    for para in paras:
        while len(para) > target_chars:
            cut = para.rfind(" ", 0, target_chars)
            if cut <= 0:
                cut = target_chars
            pieces.append(para[:cut])
            para = para[cut:].strip()
        if para:
            pieces.append(para)
    docs: list[str] = []
    buf: list[str] = []
    size = 0
    for piece in pieces:
        buf.append(piece)
        size += len(piece)
        if size >= target_chars:
            docs.append("\n\n".join(buf))
            buf, size = [], 0
    if buf:
        docs.append("\n\n".join(buf))
    return docs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the fires_on cache for one model.")
    parser.add_argument("--model", required=True,
                        help="model id, e.g. meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--sample", default="data/reader_sample.txt")
    parser.add_argument("--out-dir", default=None,
                        help="default: store/reader/<model-slug>/")
    parser.add_argument("--layers", default="all",
                        help='"all", "12-27", or "1,2,3"')
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--target-chars", type=int, default=1000,
                        help="about 250 tokens per document")
    args = parser.parse_args(argv)
    sample = Path(args.sample)
    if not sample.exists():
        print(f"sample text not found: {sample}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir or
                   str(ROOT / "store" / "reader" / slug_for(args.model)))
    docs = split_docs(sample.read_text(encoding="utf-8"),
                      target_chars=args.target_chars)
    eng = engine.get_engine()
    print(f"loading {args.model} ...", flush=True)
    eng.load(args.model)
    try:
        layers = parse_layers(args.layers, eng.spec.layers)
        start = time.perf_counter()
        reader.build_fires_on_cache(eng, docs, out_dir, layers, k=args.k)
        seconds = time.perf_counter() - start
    finally:
        eng.unload()
    saved = json.loads((out_dir / "sample.json").read_text())
    tokens = len(saved["tokens"])
    print(f"model: {args.model}")
    print(f"documents: {len(docs)}")
    print(f"tokens processed: {tokens}")
    print(f"seconds: {seconds:.1f}")
    print(f"cache: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
