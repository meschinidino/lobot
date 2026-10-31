"""One-click patch install into llama.cpp and Ollama (IN).

Drivers probe the real installed binaries (never assume), write
the GGUF/Modelfile into an install record, and run one-shot
generations for verify. All subprocess calls carry timeouts.
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from model_lobotomy import patch as patch_mod

ROOT = Path(__file__).resolve().parent.parent.parent
INSTALLS_ROOT = ROOT / "jobs" / "installs"

TARGETS = ("llamacpp", "ollama")
INSTALL_TIMEOUT = 300
GENERATE_TIMEOUT = 180
PROBE_TIMEOUT = 30

# Same markers as scripts/install_patch.py:ollama_support.
OLLAMA_SUPPORT_MARKERS = ("control-vector", "control_vector",
                          "controlvector")
# The repo's documented llama.cpp build location (see the
# gate_llamacpp.py docstring); probed, never assumed.
LLAMACPP_BUILD_CLI = ("/tmp/llama.cpp/build/bin/llama-cli")


def _run(cmd, timeout: int):
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout)


def _llama_binary() -> str | None:
    for cand in (os.environ.get("LOBOT_LLAMACPP_CLI"),
                 shutil.which("llama-cli"), LLAMACPP_BUILD_CLI):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def probe(target: str) -> dict:
    """Probe one target. Returns {target, available, version, detail}."""
    if target == "llamacpp":
        exe = _llama_binary()
        if exe is None:
            return {"target": target, "available": False,
                    "version": None,
                    "detail": "no llama-cli on PATH, in "
                    "LOBOT_LLAMACPP_CLI, or at "
                    + LLAMACPP_BUILD_CLI}
        try:
            r = _run([exe, "--version"], PROBE_TIMEOUT)
        except (OSError, subprocess.SubprocessError) as e:
            return {"target": target, "available": False,
                    "version": None,
                    "detail": f"{exe} --version failed: {e}"}
        if r.returncode != 0:
            return {"target": target, "available": False,
                    "version": None,
                    "detail": f"{exe} --version exited "
                    f"{r.returncode}: {(r.stderr or '')[-300:]}"}
        first = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
        first = first.splitlines()[0].strip() if first else "?"
        return {"target": target, "available": True,
                "version": first, "detail": exe}
    if target == "ollama":
        exe = shutil.which("ollama")
        if exe is None:
            return {"target": target, "available": False,
                    "version": None,
                    "detail": "ollama is not installed "
                    "(no 'ollama' on PATH)"}
        try:
            ver = _run([exe, "--version"], PROBE_TIMEOUT)
        except (OSError, subprocess.SubprocessError) as e:
            return {"target": target, "available": False,
                    "version": None,
                    "detail": f"ollama --version failed: {e}"}
        out = ((ver.stdout or "") + "\n" + (ver.stderr or "")).strip()
        version = out.splitlines()[0].strip() if out else "?"
        if ver.returncode != 0:
            return {"target": target, "available": False,
                    "version": version,
                    "detail": f"ollama --version exited "
                    f"{ver.returncode}"}
        help_text = ""
        for args in (["--help"], ["run", "--help"],
                     ["create", "--help"], ["show", "--help"]):
            try:
                r = _run([exe, *args], PROBE_TIMEOUT)
            except (OSError, subprocess.SubprocessError):
                continue
            help_text += ((r.stdout or "") + "\n"
                          + (r.stderr or "")).lower() + "\n"
        hits = sorted(m for m in OLLAMA_SUPPORT_MARKERS
                      if m in help_text)
        if not hits:
            return {"target": target, "available": False,
                    "version": version,
                    "detail": f"{version} shows no control-vector "
                    "support in its help output"}
        return {"target": target, "available": True,
                "version": version,
                "detail": f"{version} advertises "
                f"{', '.join(hits)} in its help output"}
    return {"target": target, "available": False, "version": None,
            "detail": f"unknown target {target!r}"}


def _patch_meta(patch_path) -> dict:
    try:
        return patch_mod.load_patch(patch_path)["meta"]
    except patch_mod.PatchRejected as e:
        raise ValueError(f"patch rejected: {e}") from None


def install(target: str, patch_path, out_dir, model_ref: str) -> dict:
    """Export + install. Returns {ok, detail, ...run record}."""
    out_dir = Path(out_dir)
    if target not in TARGETS:
        return {"ok": False, "detail": f"unknown target {target!r}"}
    if target == "llamacpp":
        pr = probe(target)
        if not pr["available"]:
            return {"ok": False, "detail": pr["detail"]}
        model = Path(str(model_ref))
        if not model.is_file():
            return {"ok": False,
                    "detail": f"model GGUF not found: {model_ref}"}
        model = model.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            exp = patch_mod.export_control_vector(
                patch_path, out_dir / "control_vector.gguf")
        except patch_mod.PatchRejected as e:
            return {"ok": False, "detail": f"export rejected: {e}"}
        base = [pr["detail"], "-m", str(model), "--temp", "0",
                "-s", "0", "--log-disable"]
        return {"ok": True, "detail": f"wrote {out_dir}",
                "binary": pr["detail"], "model": str(model),
                "gguf": str(out_dir / "control_vector.gguf"),
                "layer": exp["layer"], "strength": exp["strength"],
                "base_cmd": base, "steer_flags": list(exp["flags"])}
    pr = probe(target)
    if not pr["available"]:
        return {"ok": False, "detail": pr["detail"]}
    meta = _patch_meta(patch_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    gguf_name = "control_vector.gguf"
    try:
        patch_mod.export_control_vector(patch_path,
                                        out_dir / gguf_name)
    except patch_mod.PatchRejected as e:
        return {"ok": False, "detail": f"export rejected: {e}"}
    name = (f"lobot-{meta.get('trait', 'patch')}"
            f"-{int(time.time())}").replace(" ", "-").lower()
    snippet = (f"# lobot patch: trait {meta.get('trait')} "
               f"layer {meta.get('layer')} dose {meta.get('dose')}\n"
               f"# Probe: {pr['detail']}.\n"
               f"# Control-vector file: ./{gguf_name} "
               f"(written next to this snippet).\n"
               f"FROM {model_ref}\n")
    (out_dir / "Modelfile.lobot").write_text(snippet)
    exe = shutil.which("ollama")
    try:
        r = _run([exe, "create", name, "-f",
                  str(out_dir / "Modelfile.lobot")], INSTALL_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as e:
        return {"ok": False, "detail": f"ollama create failed: {e}"}
    if r.returncode != 0:
        return {"ok": False,
                "detail": f"ollama create exited {r.returncode}: "
                f"{(r.stderr or '')[-500:]}"}
    return {"ok": True, "detail": f"ollama model {name} created",
            "model": name,
            "gguf": str(out_dir / gguf_name)}


def _parse_llama_output(stdout: str, prompt: str) -> str:
    """Text between the prompt echo and the stats line (gate rule)."""
    lines = (stdout or "").splitlines()
    try:
        i = lines.index("> " + prompt)
    except ValueError:
        raise ValueError("prompt echo not in llama-cli output")
    gen = []
    for ln in lines[i + 1:]:
        if ln.startswith("[ Prompt:"):
            break
        if ln.strip():
            gen.append(ln)
    return "\n".join(gen)


def generate(target: str, run: dict, prompt: str,
             patched: bool = True, max_new: int = 64) -> dict:
    """One generation. Returns {ok, text, detail}."""
    if target == "llamacpp":
        cmd = [*run["base_cmd"], "-p", prompt, "-sys", "",
               "-n", str(max_new)]
        if patched:
            cmd += run["steer_flags"]
        try:
            r = _run(cmd, GENERATE_TIMEOUT)
        except subprocess.TimeoutExpired:
            return {"ok": False, "text": "",
                    "detail": "llama-cli timed out"}
        except (OSError, subprocess.SubprocessError) as e:
            return {"ok": False, "text": "",
                    "detail": f"llama-cli failed: {e}"}
        if r.returncode != 0:
            return {"ok": False, "text": "",
                    "detail": f"llama-cli exited {r.returncode}: "
                    f"{(r.stderr or '')[-500:]}"}
        try:
            return {"ok": True,
                    "text": _parse_llama_output(r.stdout, prompt),
                    "detail": "parsed"}
        except ValueError as e:
            return {"ok": False, "text": "",
                    "detail": f"{e}: {(r.stdout or '')[-300:]}"}
    if target == "ollama":
        exe = shutil.which("ollama")
        if exe is None:
            return {"ok": False, "text": "",
                    "detail": "ollama is not installed"}
        try:
            r = _run([exe, "run", run["model"], prompt],
                     GENERATE_TIMEOUT)
        except subprocess.TimeoutExpired:
            return {"ok": False, "text": "",
                    "detail": "ollama run timed out"}
        except (OSError, subprocess.SubprocessError) as e:
            return {"ok": False, "text": "",
                    "detail": f"ollama run failed: {e}"}
        if r.returncode != 0:
            return {"ok": False, "text": "",
                    "detail": f"ollama run exited {r.returncode}: "
                    f"{(r.stderr or '')[-500:]}"}
        return {"ok": True, "text": (r.stdout or "").strip(),
                "detail": "parsed"}
    return {"ok": False, "text": "",
            "detail": f"unknown target {target!r}"}


def compare(shipped_card: dict, gains: list[float]) -> dict:
    """Shipped vs measured effect. Same rule as verify_patch.compare."""
    lo, hi = shipped_card["trait_ci"]
    measured = statistics.fmean(gains)
    return {"shipped_mean": shipped_card["trait_mean"],
            "shipped_ci": [lo, hi], "measured_mean": measured,
            "n": len(gains), "pass": bool(lo <= measured <= hi)}


def save_install(record: dict) -> Path:
    """Write jobs/installs/<id>/install.json. Returns the dir."""
    d = INSTALLS_ROOT / record["install_id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "install.json").write_text(json.dumps(record, indent=2))
    return d


def load_install(install_id: str) -> dict | None:
    """Read one install record, or None."""
    p = INSTALLS_ROOT / install_id / "install.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())
