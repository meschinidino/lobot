"""Install a .lobot patch into llama.cpp or Ollama.

llamacpp: writes the control-vector GGUF (via export_control_vector)
and prints the exact llama-cli flags.
ollama: writes a Modelfile snippet only when the installed ollama
binary shows control-vector support; else it says so and exits 2.

Run:
    uv run python scripts/install_patch.py knob.lobot --target llamacpp --out DIR
    uv run python scripts/install_patch.py knob.lobot --target ollama --out DIR
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import patch as patch_mod  # noqa: E402

# Markers that count as control-vector support in ollama help text.
OLLAMA_SUPPORT_MARKERS = ("control-vector", "control_vector",
                          "controlvector")


def ollama_support(ollama_exe: str | None = None) -> tuple[bool, str]:
    """Probe the installed ollama binary for control-vector support.

    Returns (supported, detail). The check reads the installed
    binary's own --version and help output; it never assumes a
    version supports control vectors.
    """
    exe = ollama_exe or shutil.which("ollama")
    if exe is None:
        return (False, "ollama is not installed (no 'ollama' on PATH)")
    try:
        ver = subprocess.run([exe, "--version"], capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        return (False, f"ollama --version failed: {e}")
    version = (ver.stdout + ver.stderr).strip().splitlines()
    version = version[0].strip() if version else "unknown version"
    help_text = ""
    for args in (["--help"], ["run", "--help"], ["create", "--help"],
                 ["show", "--help"]):
        try:
            r = subprocess.run([exe, *args], capture_output=True,
                               text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            continue
        help_text += (r.stdout + "\n" + r.stderr).lower() + "\n"
    hits = sorted(m for m in OLLAMA_SUPPORT_MARKERS if m in help_text)
    if hits:
        return (True, f"{version} advertises {', '.join(hits)} "
                f"in its help output")
    return (False,
            f"{version} shows no control-vector support in its help "
            f"output (looked for {', '.join(OLLAMA_SUPPORT_MARKERS)})")


def install_llamacpp(patch_path, out_dir: Path) -> dict:
    """Write the control-vector GGUF into out_dir. Returns the export dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_gguf = out_dir / "control_vector.gguf"
    return patch_mod.export_control_vector(patch_path, out_gguf)


def install_ollama(patch_path, out_dir: Path,
                   ollama_exe: str | None = None) -> tuple[int, str]:
    """Write a Modelfile snippet, or return (2, reason) when unsupported."""
    supported, detail = ollama_support(ollama_exe)
    if not supported:
        return (2, f"ollama target unsupported: {detail}; "
                f"no Modelfile written")
    meta = patch_mod.load_patch(patch_path)["meta"]
    out_dir.mkdir(parents=True, exist_ok=True)
    gguf_name = "control_vector.gguf"
    snippet = (f"# lobot patch: trait {meta.get('trait')} "
               f"layer {meta.get('layer')} dose {meta.get('dose')}\n"
               f"# Probe: {detail}.\n"
               f"# Control-vector file: ./{gguf_name} "
               f"(written next to this snippet).\n"
               f"# Finish with the control-vector directive your "
               f"ollama help shows, then `ollama create`.\n"
               f"FROM {meta.get('trait', 'model')}\n")
    (out_dir / "Modelfile.lobot").write_text(snippet)
    patch_mod.export_control_vector(patch_path, out_dir / gguf_name)
    return (0, detail)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("patch", help=".lobot patch file")
    ap.add_argument("--target", required=True,
                    choices=("llamacpp", "ollama"))
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    if args.target == "llamacpp":
        try:
            got = install_llamacpp(args.patch, out_dir)
        except patch_mod.PatchRejected as e:
            print(f"rejected: {e}", flush=True)
            return 2
        print(f"wrote {out_dir / 'control_vector.gguf'} "
              f"(direction.{got['layer']}, strength {got['strength']})",
              flush=True)
        print("llama-cli flags:", " ".join(got["flags"]), flush=True)
        return 0
    try:
        code, detail = install_ollama(args.patch, out_dir)
    except patch_mod.PatchRejected as e:
        print(f"rejected: {e}", flush=True)
        return 2
    if code == 0:
        print(f"wrote {out_dir / 'Modelfile.lobot'} ({detail})",
              flush=True)
        return 0
    print(detail, flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
