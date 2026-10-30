"""Wave-0 gate: llama.cpp control-vector round-trip vs lobot output.

Exports the frozen trait knob (logs/e2e_<trait>/knob.lobot) to a
llama.cpp control-vector GGUF and checks that llama-cli steered with
it produces the same top tokens as Engine.run steered with the same
vector. A layer off-by-one on either side scrambles early tokens, so
the gate compares the first-8-token prefix: PASS when lobot and
llama.cpp agree on all 8 steered tokens AND on all 8 baseline tokens
(the baseline pair validates the harness itself).

Prereqs (run once, AFTER any E2E: conversion loads full weights):
    cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build -DGGML_METAL=ON
    cmake --build /tmp/llama.cpp/build -j --target llama-cli
The script converts the F16 GGUF itself if missing
(/tmp/llama.cpp/convert_hf_to_gguf.py from the HF cache).

Writes logs/gate_llamacpp/report.json + report.md. Exit 0/1.

Run:
    uv run python scripts/gate_llamacpp.py [--trait blunt]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import engine as engine_mod  # noqa: E402
from model_lobotomy import patch as patch_mod  # noqa: E402
from model_lobotomy import steer  # noqa: E402

LLAMACPP = Path("/tmp/llama.cpp")
PROMPT = "Explain how photosynthesis works in one paragraph."
MAX_NEW = 32
PREFIX_TOKENS = 8


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=steer.LLAMA_MODEL_ID)
    ap.add_argument("--trait", default="blunt")
    ap.add_argument("--workdir",
                    default=str(ROOT / "logs" / "gate_llamacpp"))
    args = ap.parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out: dict = {"gate": "llamacpp", "model": args.model,
                 "trait": args.trait, "pass": False}

    def fail(reason):
        out["fail_reason"] = reason
        (workdir / "report.json").write_text(json.dumps(out, indent=2))
        (workdir / "report.md").write_text(
            f"# Gate llama.cpp: FAIL\n{reason}\n")
        print(f"[gate-llamacpp] FAIL: {reason}", flush=True)
        return 1

    cli = LLAMACPP / "build" / "bin" / "llama-cli"
    if not cli.exists():
        return fail(f"{cli} missing; build llama.cpp first (see docstring)")
    knob = ROOT / "logs" / f"e2e_{args.trait}" / "knob.lobot"
    if not knob.exists():
        return fail(f"{knob} missing; run the trait E2E first")
    gguf_model = workdir / "model-f16.gguf"
    if not gguf_model.exists():
        print("[gate-llamacpp] converting F16 GGUF (slow, ~GBs of RAM)",
              flush=True)
        from huggingface_hub import snapshot_download
        src = snapshot_download(args.model)
        r = _run([sys.executable, str(LLAMACPP / "convert_hf_to_gguf.py"),
                  src, "--outfile", str(gguf_model),
                  "--outtype", "f16"], cwd=str(ROOT))
        if r.returncode != 0 or not gguf_model.exists():
            return fail(f"GGUF conversion failed: {r.stderr[-2000:]}")
    cvec = workdir / "control_vector.gguf"
    try:
        exp = patch_mod.export_control_vector(knob, cvec)
    except patch_mod.PatchRejected as e:
        return fail(f"export rejected: {e}")

    eng = engine_mod.Engine()
    eng.load(args.model)
    print(f"[gate-llamacpp] {args.model} warm "
          f"(rev {eng.revision[:12]})", flush=True)
    try:
        spec = eng.spec
        layer = exp["layer"]
        strength = exp["strength"]
        meta = patch_mod.load_patch(knob)["meta"]
        # Both stacks steer with the patch scale (dose * scale * unit);
        # medians are unused on the vector-only hook path. No fresh
        # calibration: medians depend on the prompt set, so a new one
        # would not match the patch scale by construction.
        unit = tuple(patch_mod.load_patch(knob)["direction"].tolist())
        edits = engine_mod.EditSet(vector=(engine_mod.VectorEdit(
            layer, unit, float(meta["scale"]),
            float(meta["dose"])),))
        base = eng.run(PROMPT, engine_mod.EMPTY_EDITS, medians={},
                       max_new_tokens=MAX_NEW, seed=0)
        steered = eng.run(PROMPT, edits, medians={},
                          max_new_tokens=MAX_NEW, seed=0)

        def llama_run(extra):
            # Raw user text + empty system prompt: llama-cli applies
            # its chat template (user/assistant headers, no system),
            # which the harness prefix check verifies against ours.
            # Parsed from stdout: lines after the "> <prompt>" echo
            # up to the "[ Prompt:" stats line.
            r = _run([str(cli), "-m", str(gguf_model), "-p", PROMPT,
                      "-sys", "", "-n", str(MAX_NEW), "--temp", "0",
                      "-s", "0", "--log-disable", *extra],
                     cwd=str(workdir))
            if r.returncode != 0:
                raise RuntimeError(f"llama-cli failed: {r.stderr[-1000:]}")
            lines = r.stdout.splitlines()
            try:
                i = lines.index("> " + PROMPT)
            except ValueError:
                raise RuntimeError(
                    f"prompt echo not in llama-cli output: "
                    f"{r.stdout[-500:]}")
            gen = []
            for ln in lines[i + 1:]:
                if ln.startswith("[ Prompt:"):
                    break
                if ln.strip():
                    gen.append(ln)
            return "\n".join(gen)

        cpp_base_text = llama_run([])
        cpp_steered_text = llama_run(exp["flags"])
        tok = eng.tokenizer
        cpp_base_ids = tok(cpp_base_text,
                           add_special_tokens=False)["input_ids"]
        cpp_steered_ids = tok(cpp_steered_text,
                              add_special_tokens=False)["input_ids"]
        div_base = _first_divergence(
            base["token_ids"][:PREFIX_TOKENS],
            cpp_base_ids[:PREFIX_TOKENS])
        div_steered = _first_divergence(
            steered["token_ids"][:PREFIX_TOKENS],
            cpp_steered_ids[:PREFIX_TOKENS])
        harness_ok = div_base is None
        roundtrip_ok = div_steered is None
        out.update({
            "fingerprint": eng.fingerprint(),
            "strength": strength, "layer": layer,
            "harness_prefix_match": harness_ok,
            "harness_divergence": div_base,
            "steered_prefix_match": roundtrip_ok,
            "steered_divergence": div_steered,
            "prefix_tokens": PREFIX_TOKENS,
            "lobot_steered_ids": steered["token_ids"][:PREFIX_TOKENS],
            "cpp_steered_ids": cpp_steered_ids[:PREFIX_TOKENS],
            "pass": bool(harness_ok and roundtrip_ok),
        })
    finally:
        eng.unload()
    (workdir / "report.json").write_text(json.dumps(out, indent=2))
    (workdir / "report.md").write_text(
        f"# Gate llama.cpp: {'PASS' if out['pass'] else 'FAIL'}\n"
        f"harness (baseline) prefix-{PREFIX_TOKENS} match: "
        f"{out.get('harness_prefix_match')} "
        f"(divergence {out.get('harness_divergence')})\n"
        f"steered prefix-{PREFIX_TOKENS} match: "
        f"{out.get('steered_prefix_match')} "
        f"(divergence {out.get('steered_divergence')})\n"
        f"layer {out.get('layer')} strength {out.get('strength')}\n")
    print(f"[gate-llamacpp] {'PASS' if out['pass'] else 'FAIL'}",
          flush=True)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
