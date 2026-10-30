"""S3-1 .lobot patch: zip of safetensors directions + JSON metadata.

Patches are made to be shared, so imports never unpickle: loading
rejects any archive carrying a pickled payload (.pkl/.pickle/.pt/
.bin) or a version it does not understand, before touching tensors.

Layout inside the zip:
    direction.safetensors   {"direction": float tensor}
    patch.json              {version, trait, layer, dose, recipe,
                             report_card, guard, fingerprint,
                             evidence_sha256, members_sha256,
                             lobot_version}
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import torch

PATCH_VERSION = "lobot-patch-v1"

# Keep in sync with pyproject.toml.
LOBOT_VERSION = "0.1.0"

DIRECTION_FILE = "direction.safetensors"
META_FILE = "patch.json"

# Rejected on sight: anything torch.load could execute.
PICKLE_SUFFIXES = (".pkl", ".pickle", ".pt", ".pth", ".bin")


class PatchRejected(ValueError):
    """Raised when a .lobot archive fails validation (never loaded)."""


def canonical_json(obj) -> bytes:
    """Canonical bytes of a JSON value: sorted keys, tight separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_patch(path, *, direction, layer: int, dose: float,
               trait: str, recipe: dict, report_card: dict,
               guard: dict, fingerprint: dict,
               scale: float | None = None) -> Path:
    """Write a .lobot patch (safetensors + JSON, no pickle anywhere).

    scale is the layer median norm: the shipped steering vector is
    dose * scale * direction (llama.cpp: --control-vector-scaled with
    strength dose * scale).
    """
    from safetensors.torch import save as _save
    path = Path(path)
    direction_bytes = _save({"direction": direction.detach().cpu()})
    meta = {"version": PATCH_VERSION, "trait": trait,
            "layer": int(layer), "dose": float(dose),
            "scale": None if scale is None else float(scale),
            "direction": {"shape": list(direction.shape),
                          "dtype": str(direction.dtype).replace("torch.", "")},
            "recipe": recipe, "report_card": report_card,
            "guard": guard, "fingerprint": fingerprint,
            "evidence_sha256": sha256_hex(canonical_json(report_card)),
            "members_sha256": {DIRECTION_FILE: sha256_hex(direction_bytes)},
            "lobot_version": LOBOT_VERSION}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(DIRECTION_FILE, direction_bytes)
        zf.writestr(META_FILE, json.dumps(meta, indent=2))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return path


def load_patch(path) -> dict:
    """Validate and load a .lobot patch. Never unpickles.

    Returns {"direction": tensor, "meta": patch.json dict}.
    Raises PatchRejected for hostile or malformed archives, and when
    the saved hashes do not match the members or the evidence.
    Old patches without hashes load with meta["unverified"] True.
    """
    from safetensors.torch import load as _load
    path = Path(path)
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        raise PatchRejected(f"not a zip archive: {path}") from e
    with zf:
        names = zf.namelist()
        for n in names:
            low = n.lower()
            if low.endswith(PICKLE_SUFFIXES):
                raise PatchRejected(
                    f"pickled payload rejected: {n} (patches must be "
                    f"safetensors + JSON only)")
        if META_FILE not in names:
            raise PatchRejected(f"missing {META_FILE}")
        if DIRECTION_FILE not in names:
            raise PatchRejected(f"missing {DIRECTION_FILE}")
        try:
            meta = json.loads(zf.read(META_FILE))
        except json.JSONDecodeError as e:
            raise PatchRejected(f"{META_FILE} is not JSON: {e}") from e
        if meta.get("version") != PATCH_VERSION:
            raise PatchRejected(
                f"unsupported patch version {meta.get('version')!r}; "
                f"want {PATCH_VERSION!r}")
        members = meta.get("members_sha256")
        evidence = meta.get("evidence_sha256")
        direction_bytes = zf.read(DIRECTION_FILE)
        if isinstance(members, dict) and evidence is not None:
            want_member = members.get(DIRECTION_FILE)
            if want_member != sha256_hex(direction_bytes):
                raise PatchRejected(
                    f"{DIRECTION_FILE} hash mismatch: patch was changed "
                    f"after it was saved")
            if evidence != sha256_hex(canonical_json(
                    meta.get("report_card"))):
                raise PatchRejected(
                    "report_card hash mismatch: evidence was changed "
                    "after the patch was saved")
            meta["unverified"] = False
        else:
            # Old patch without hashes: loads, flagged as unverified.
            meta["unverified"] = True
        try:
            tensors = _load(direction_bytes)
        except Exception as e:
            raise PatchRejected(
                f"{DIRECTION_FILE} is not valid safetensors: {e}") from e
        if "direction" not in tensors:
            raise PatchRejected(f"{DIRECTION_FILE} lacks 'direction'")
        direction = tensors["direction"]
        want_shape = (meta.get("direction") or {}).get("shape")
        if want_shape is not None and list(direction.shape) != want_shape:
            raise PatchRejected(
                f"direction shape {list(direction.shape)} != meta "
                f"{want_shape}")
    return {"direction": direction, "meta": meta}


def export_control_vector(patch_path, out_gguf) -> dict:
    """Export a .lobot patch to a llama.cpp control-vector GGUF.

    Writes one F32 1-D tensor `direction.{L}` (L = 0-based frozen
    layer; llama.cpp forbids layer 0 and applies direction.{k} at
    0-based layer k, after the FFN residual add — the same site as
    our vector edits). Returns {"strength", "layer", "flags"} where
    flags are the llama-cli arguments
    (--control-vector-scaled <gguf>:<strength>
    --control-vector-layer-range L L) with strength = dose * scale.
    """
    loaded = load_patch(patch_path)
    meta = loaded["meta"]
    layer = int(meta["layer"])
    if layer < 1:
        raise PatchRejected(
            f"layer {layer} cannot export: llama.cpp forbids "
            f"direction.0 (no control vector for layer 0)")
    if meta.get("scale") is None:
        raise PatchRejected("patch has no scale; cannot set strength")
    import gguf
    vec = loaded["direction"].detach().cpu().to(torch.float32).numpy()
    out_gguf = Path(out_gguf)
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out_gguf), "llama")
    writer.add_tensor(f"direction.{layer}", vec)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    strength = float(meta["dose"]) * float(meta["scale"])
    return {"strength": strength, "layer": layer,
            "flags": ["--control-vector-scaled",
                      f"{out_gguf}:{strength}",
                      "--control-vector-layer-range",
                      str(layer), str(layer)]}
