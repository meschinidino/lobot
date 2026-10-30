"""Export a .lobot patch to a llama.cpp control-vector GGUF.

Reads the patch (direction + layer + dose + scale), writes one F32
1-D tensor `direction.{L}`, and prints the llama-cli flags
(--control-vector-scaled <gguf>:<dose*scale>
--control-vector-layer-range L L).

Run:
    uv run python scripts/export_llamacpp.py jobs/<id>/trait/knob.lobot -o control_vector.gguf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import patch as patch_mod  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("patch", help=".lobot patch file")
    ap.add_argument("-o", "--out", default="control_vector.gguf")
    args = ap.parse_args()
    try:
        got = patch_mod.export_control_vector(args.patch, args.out)
    except patch_mod.PatchRejected as e:
        print(f"rejected: {e}", flush=True)
        return 2
    print(f"wrote {args.out} (direction.{got['layer']}, "
          f"strength {got['strength']})", flush=True)
    print("llama-cli flags:", " ".join(got["flags"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
