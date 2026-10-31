"""Signed patch registry (1000X RG): publish, verify, install.

A registry is a plain directory (a git repo or any static file host
works): index.json, patches/<sha256>.lobot and keys/<key_id>.pub.
Each entry carries an ed25519 signature over a canonical statement
that binds the patch bytes (sha256), the trait, the model fingerprint
and the evidence hash. Install re-checks every link and fails loudly:
a changed patch, a changed index entry, an unknown or untrusted key,
or a bad signature each raise RegistryError.

Dual use (roadmap later.md): publish refuses patches without a guard
record, patches the guard blocked or would block (refusal-weakening),
patches made in research mode, and old patches without hashes.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey)

from model_lobotomy import audit
from model_lobotomy import patch as patch_mod

INDEX_VERSION = "lobot-registry-v1"
STATEMENT_KIND = "lobot-publish-v1"
INDEX_FILE = "index.json"
HOME = Path(os.environ.get("LOBOT_HOME", Path.home() / ".lobot"))

# Statement fields, in the order they are documented; the signature
# covers canonical_json of exactly these keys.
STATEMENT_KEYS = ("kind", "name", "patch_sha256", "trait", "model_id",
                  "revision", "evidence_sha256", "published_at",
                  "key_id")


class RegistryError(RuntimeError):
    """A publish was refused or an entry failed verification."""


# -- keys -----------------------------------------------------------------------


def key_id_of(public_raw: bytes) -> str:
    """First 16 hex chars of sha256(raw public key)."""
    return patch_mod.sha256_hex(public_raw)[:16]


def _public_raw(key: Ed25519PrivateKey | Ed25519PublicKey) -> bytes:
    pub = key.public_key() if isinstance(key, Ed25519PrivateKey) else key
    return pub.public_bytes(serialization.Encoding.Raw,
                            serialization.PublicFormat.Raw)


def generate_key(keys_dir: Path | None = None) -> dict:
    """New ed25519 signing key, saved 0600. Returns {key_id, path}."""
    keys_dir = Path(keys_dir or HOME / "keys")
    keys_dir.mkdir(parents=True, exist_ok=True)
    key = Ed25519PrivateKey.generate()
    kid = key_id_of(_public_raw(key))
    path = keys_dir / f"{kid}.pem"
    pem = key.private_bytes(serialization.Encoding.PEM,
                            serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(pem)
    return {"key_id": kid, "path": str(path),
            "public_key": _public_raw(key).hex()}


def load_key(path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(Path(path).read_bytes(),
                                             password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise RegistryError(f"{path} is not an ed25519 key")
    return key


def load_trusted(path: Path | None = None) -> set[str]:
    """Trusted publisher key ids (empty when the file is missing)."""
    path = Path(path or HOME / "trusted.json")
    if not path.is_file():
        return set()
    return set(json.loads(path.read_text()))


def trust(key_id: str, path: Path | None = None) -> set[str]:
    """Add a publisher key id to the trust list. Returns the list."""
    path = Path(path or HOME / "trusted.json")
    ids = load_trusted(path) | {key_id}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(ids), indent=2))
    return ids


# -- index ------------------------------------------------------------------------


def _index_path(registry) -> Path:
    return Path(registry) / INDEX_FILE


def load_index(registry) -> dict:
    path = _index_path(registry)
    if not path.is_file():
        return {"version": INDEX_VERSION, "entries": []}
    data = json.loads(path.read_text())
    if data.get("version") != INDEX_VERSION:
        raise RegistryError(f"unsupported registry version "
                            f"{data.get('version')!r}")
    return data


def _save_index(registry, data: dict) -> None:
    path = _index_path(registry)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def statement_of(entry: dict) -> bytes:
    """The signed bytes for an entry: canonical JSON of STATEMENT_KEYS."""
    return patch_mod.canonical_json({k: entry.get(k)
                                     for k in STATEMENT_KEYS})


# -- publish ----------------------------------------------------------------------


def check_publishable(meta: dict) -> None:
    """Dual-use and evidence rules. Raises RegistryError when refused."""
    if meta.get("unverified"):
        raise RegistryError("patch has no member/evidence hashes; "
                            "re-export it before publishing")
    guard = meta.get("guard") or {}
    if not guard:
        raise RegistryError("patch has no guard record; run the guard "
                            "before publishing")
    if guard.get("research_mode"):
        raise RegistryError("patch was made in research mode; research "
                            "patches cannot be published")
    if guard.get("blocked") or guard.get("would_block"):
        if not guard.get("verified"):
            raise RegistryError(
                "the guard flagged this patch from the prefill signal "
                "only (not confirmed); run `lobot_registry.py guard "
                "PATCH` to confirm it with full generation")
        flagged = sorted(n for n, sc in (guard.get("screens") or {})
                         .items() if sc.get("blocked")
                         or sc.get("would_block")) or ["refusal"]
        raise RegistryError(f"the guard flags this patch "
                            f"({', '.join(flagged)}, confirmed by full "
                            f"generation); it cannot be published")
    if not guard.get("verified"):
        raise RegistryError("the guard verdict is not confirmed by full "
                            "generation; run `lobot_registry.py guard "
                            "PATCH` first")
    card = meta.get("report_card") or {}
    if "trait_mean" not in card or "trait_ci" not in card:
        raise RegistryError("patch has no shipped evidence "
                            "(report_card trait_mean, trait_ci)")
    if not (meta.get("fingerprint") or {}).get("model_id"):
        raise RegistryError("patch names no model_id in its fingerprint")


def publish(registry, patch_path, key: Ed25519PrivateKey, *,
            name: str | None = None, audit_log=None) -> dict:
    """Sign a patch into the registry. Returns the new index entry."""
    registry = Path(registry)
    data = Path(patch_path).read_bytes()
    try:
        meta = patch_mod.load_patch(patch_path)["meta"]
    except patch_mod.PatchRejected as e:
        raise RegistryError(f"patch rejected: {e}") from None
    check_publishable(meta)
    sha = patch_mod.sha256_hex(data)
    index = load_index(registry)
    if any(e["patch_sha256"] == sha for e in index["entries"]):
        raise RegistryError(f"patch {sha[:12]} is already published")
    fp = meta.get("fingerprint") or {}
    name = name or f"{meta.get('trait')}@{fp.get('model_id')}"
    if any(e["name"] == name for e in index["entries"]):
        raise RegistryError(f"name {name!r} is taken; pass another")
    raw_pub = _public_raw(key)
    entry = {"kind": STATEMENT_KIND, "name": name, "patch_sha256": sha,
             "trait": meta.get("trait"),
             "model_id": fp.get("model_id"),
             "revision": fp.get("revision"),
             "evidence_sha256": meta.get("evidence_sha256"),
             "published_at": round(time.time(), 3),
             "key_id": key_id_of(raw_pub)}
    entry["signature"] = key.sign(statement_of(entry)).hex()
    entry["public_key"] = raw_pub.hex()
    card = meta.get("report_card") or {}
    entry["evidence"] = {"trait_mean": card.get("trait_mean"),
                         "trait_ci": card.get("trait_ci"),
                         "passed": card.get("passed")}
    (registry / "patches").mkdir(parents=True, exist_ok=True)
    (registry / "keys").mkdir(parents=True, exist_ok=True)
    (registry / "patches" / f"{sha}.lobot").write_bytes(data)
    (registry / "keys" / f"{entry['key_id']}.pub").write_text(
        raw_pub.hex())
    index["entries"].append(entry)
    _save_index(registry, index)
    audit.record("publish", log=audit_log, name=name, patch_sha256=sha,
                 key_id=entry["key_id"], registry=str(registry))
    return entry


# -- verify / install ---------------------------------------------------------------


def find(registry, ref: str) -> dict:
    """Entry by name or patch sha256 (prefix of 8+ chars)."""
    hits = [e for e in load_index(registry)["entries"]
            if e["name"] == ref
            or (len(ref) >= 8 and e["patch_sha256"].startswith(ref))]
    if len(hits) != 1:
        raise RegistryError(f"{len(hits)} entries match {ref!r}")
    return hits[0]


def verify_entry(registry, entry: dict,
                 trusted: set[str] | None) -> dict:
    """Check one entry end to end. Returns the loaded patch.

    trusted None skips the trust check (listing, auditing); install
    always passes a set.
    """
    registry = Path(registry)
    pub_hex = entry.get("public_key") or ""
    try:
        raw_pub = bytes.fromhex(pub_hex)
        pub = Ed25519PublicKey.from_public_bytes(raw_pub)
    except ValueError:
        raise RegistryError(f"{entry.get('name')}: bad public key") \
            from None
    if key_id_of(raw_pub) != entry.get("key_id"):
        raise RegistryError(f"{entry.get('name')}: key id does not "
                            f"match the public key")
    key_file = registry / "keys" / f"{entry['key_id']}.pub"
    if not key_file.is_file() or key_file.read_text().strip() != pub_hex:
        raise RegistryError(f"{entry.get('name')}: publisher key file "
                            f"missing or changed")
    try:
        pub.verify(bytes.fromhex(entry.get("signature") or ""),
                   statement_of(entry))
    except (InvalidSignature, ValueError):
        raise RegistryError(f"{entry.get('name')}: signature does not "
                            f"match the entry (index changed?)") from None
    if trusted is not None and entry["key_id"] not in trusted:
        raise RegistryError(f"{entry.get('name')}: publisher "
                            f"{entry['key_id']} is not trusted; run "
                            f"`trust {entry['key_id']}` if you know them")
    path = registry / "patches" / f"{entry['patch_sha256']}.lobot"
    if not path.is_file():
        raise RegistryError(f"{entry.get('name')}: patch file missing")
    if patch_mod.sha256_hex(path.read_bytes()) != entry["patch_sha256"]:
        raise RegistryError(f"{entry.get('name')}: patch bytes changed "
                            f"after signing")
    try:
        loaded = patch_mod.load_patch(path)
    except patch_mod.PatchRejected as e:
        raise RegistryError(f"{entry.get('name')}: {e}") from None
    meta = loaded["meta"]
    fp = meta.get("fingerprint") or {}
    claims = {"trait": meta.get("trait"), "model_id": fp.get("model_id"),
              "revision": fp.get("revision"),
              "evidence_sha256": meta.get("evidence_sha256")}
    for k, v in claims.items():
        if entry.get(k) != v:
            raise RegistryError(f"{entry.get('name')}: entry {k} "
                                f"{entry.get(k)!r} != patch {v!r}")
    return loaded


def install(registry, ref: str, dest_dir, *,
            trusted: set[str] | None = None, audit_log=None) -> dict:
    """Verify, then copy the patch into dest_dir. Returns a record."""
    entry = find(registry, ref)
    trusted = load_trusted() if trusted is None else trusted
    verify_entry(registry, entry, trusted)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{entry['patch_sha256'][:16]}.lobot"
    shutil.copyfile(Path(registry) / "patches"
                    / f"{entry['patch_sha256']}.lobot", dest)
    audit.record("install", log=audit_log, name=entry["name"],
                 patch_sha256=entry["patch_sha256"],
                 key_id=entry["key_id"], model_id=entry["model_id"],
                 path=str(dest))
    return {"name": entry["name"], "path": str(dest),
            "patch_sha256": entry["patch_sha256"],
            "model_id": entry["model_id"], "key_id": entry["key_id"]}


def uninstall(path, *, audit_log=None) -> None:
    """Remove an installed patch file and record it."""
    path = Path(path)
    sha = patch_mod.sha256_hex(path.read_bytes())
    path.unlink()
    audit.record("uninstall", log=audit_log, patch_sha256=sha,
                 path=str(path))
