"""S3-1 .lobot patch: zip of safetensors directions + JSON metadata.

Patches are made to be shared, so imports never unpickle: loading
rejects any archive carrying a pickled payload (.pkl/.pickle/.pt/
.bin) or a version it does not understand, before touching tensors.

Layout inside the zip:
    direction.safetensors   {"direction": float tensor}
    patch.json              {version, trait, layer, dose, recipe,
                             report_card, guard, fingerprint}
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

PATCH_VERSION = "lobot-patch-v1"

DIRECTION_FILE = "direction.safetensors"
META_FILE = "patch.json"

# Rejected on sight: anything torch.load could execute.
PICKLE_SUFFIXES = (".pkl", ".pickle", ".pt", ".pth", ".bin")


class PatchRejected(ValueError):
    """Raised when a .lobot archive fails validation (never loaded)."""


def save_patch(path, *, direction, layer: int, dose: float,
               trait: str, recipe: dict, report_card: dict,
               guard: dict, fingerprint: dict) -> Path:
    """Write a .lobot patch (safetensors + JSON, no pickle anywhere)."""
    from safetensors.torch import save as _save
    path = Path(path)
    meta = {"version": PATCH_VERSION, "trait": trait,
            "layer": int(layer), "dose": float(dose),
            "direction": {"shape": list(direction.shape),
                          "dtype": str(direction.dtype).replace("torch.", "")},
            "recipe": recipe, "report_card": report_card,
            "guard": guard, "fingerprint": fingerprint}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(DIRECTION_FILE,
                    _save({"direction": direction.detach().cpu()}))
        zf.writestr(META_FILE, json.dumps(meta, indent=2))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return path


def load_patch(path) -> dict:
    """Validate and load a .lobot patch. Never unpickles.

    Returns {"direction": tensor, "meta": patch.json dict}.
    Raises PatchRejected for hostile or malformed archives.
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
        try:
            tensors = _load(zf.read(DIRECTION_FILE))
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
