"""Convert a pinned HF snapshot to GGUF for the llama backend (D3).

Resolves the revision exactly like Engine.load, converts the cached
snapshot with the pinned third_party/llama.cpp converter (matching
gguf-py via PYTHONPATH, never the env package), optionally quantizes
with the pinned llama-quantize, and writes
models/<slug>/<rev12>-<quant>.gguf plus a .json sidecar (HF repo,
revision, llamacpp commit, quant, sha256).

Qwen3.5 converts as text-only (no mmproj): the product never feeds
images, and the daemon loads one text GGUF.

Run:
    uv run python scripts/convert_gguf.py --model meta-llama/Llama-3.2-3B-Instruct --quant f16
    uv run python scripts/convert_gguf.py --model Qwen/Qwen3.5-4B --quant q8_0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import llama_backend as lb  # noqa: E402
from model_lobotomy import steer  # noqa: E402

QUANTS = {"f16": "F16", "q8_0": "Q8_0", "q4_k_m": "Q4_K_M"}


def _run(cmd, env=None) -> None:
    print(f"[convert-gguf] $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, env=env or os.environ.copy())
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode})")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model id (must be in steer.MODEL_SPECS)")
    ap.add_argument("--quant", default="f16", choices=sorted(QUANTS),
                    help="output quant tag (default f16)")
    ap.add_argument("--revision", default=None,
                    help="pin a revision (default: current main sha, "
                    "resolved like Engine.load)")
    ap.add_argument("--force", action="store_true",
                    help="reconvert even when the GGUF exists")
    args = ap.parse_args()

    spec = steer.get_spec(args.model)  # loud on unknown ids
    _ = spec
    from huggingface_hub import HfApi, snapshot_download
    sha = args.revision or HfApi().model_info(args.model).sha
    print(f"[convert-gguf] {args.model} rev {sha[:12]}", flush=True)
    snap = Path(snapshot_download(args.model, revision=sha))

    lock = (ROOT / "scripts" / "llamacpp.lock").read_text().strip()
    src = ROOT / "third_party" / "llama.cpp"
    if not (src / "convert_hf_to_gguf.py").is_file():
        print("pinned toolchain missing; run "
              "scripts/build_llamacpp.sh first", flush=True)
        return 2

    out = (ROOT / "models" / lb.model_slug(args.model)
           / f"{sha[:12]}-{args.quant}.gguf")
    sidecar_path = out.with_name(out.stem + ".json")
    if out.is_file() and sidecar_path.is_file() and not args.force:
        print(f"[convert-gguf] exists: {out}", flush=True)
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = (str(src / "gguf-py") + os.pathsep
                         + env.get("PYTHONPATH", ""))
    tmp_f16 = out.with_name(out.stem + ".tmp-f16.gguf")
    _run([sys.executable, str(src / "convert_hf_to_gguf.py"),
          str(snap), "--outfile", str(tmp_f16),
          "--outtype", "f16"], env=env)
    if args.quant == "f16":
        tmp_f16.rename(out)
    else:
        quantize = (src / "build" / "bin" / "llama-quantize")
        if not quantize.is_file():
            print("llama-quantize missing; run "
                  "scripts/build_llamacpp.sh first", flush=True)
            return 2
        _run([str(quantize), str(tmp_f16), str(out),
              QUANTS[args.quant]])
        tmp_f16.unlink()
    sidecar = {"hf_repo": args.model, "revision": sha,
               "llamacpp_commit": lock, "quant": args.quant,
               "gguf_sha256": _sha256(out)}
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"[convert-gguf] wrote {out}", flush=True)
    print(f"[convert-gguf] sidecar {sidecar_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
