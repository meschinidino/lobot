"""Audit trail (1000X AU): an append-only, hash-chained event log.

Every change to what a model runs with (publish, install, uninstall,
verify) appends one JSON line. Each line carries seq, the previous
line's hash and its own hash over canonical JSON, so an auditor can
prove the log was not edited, cut or reordered (verify_chain) and
rebuild which patches were active, with their proofs (history).

The log path is LOBOT_AUDIT_LOG, else ~/.lobot/audit.jsonl.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from model_lobotomy import patch as patch_mod

GENESIS = "0" * 64


def log_path(log=None) -> Path:
    if log is not None:
        return Path(log)
    env = os.environ.get("LOBOT_AUDIT_LOG")
    if env:
        return Path(env)
    return Path(os.environ.get("LOBOT_HOME", Path.home() / ".lobot")) \
        / "audit.jsonl"


def _line_hash(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return patch_mod.sha256_hex(patch_mod.canonical_json(body))


def _last(path: Path) -> dict | None:
    if not path.is_file():
        return None
    last = None
    with path.open() as f:
        for line in f:
            if line.strip():
                last = line
    return json.loads(last) if last else None


def record(event: str, *, log=None, **fields) -> dict:
    """Append one event. Returns the written entry (with its hash)."""
    path = log_path(log)
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = _last(path)
    entry = {"seq": 0 if prev is None else prev["seq"] + 1,
             "at": round(time.time(), 3), "event": event,
             "prev": GENESIS if prev is None else prev["hash"],
             **fields}
    entry["hash"] = _line_hash(entry)
    with path.open("a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def read(log=None) -> list[dict]:
    path = log_path(log)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def verify_chain(log=None) -> dict:
    """{ok, n, bad_seq, reason}: every link, seq and hash checked."""
    prev_hash, n = GENESIS, 0
    for i, entry in enumerate(read(log)):
        reason = None
        if entry.get("seq") != i:
            reason = f"seq {entry.get('seq')} at line {i}"
        elif entry.get("prev") != prev_hash:
            reason = "prev hash does not match the line before"
        elif entry.get("hash") != _line_hash(entry):
            reason = "line hash does not match its content"
        if reason:
            return {"ok": False, "n": i, "bad_seq": i, "reason": reason}
        prev_hash, n = entry["hash"], i + 1
    return {"ok": True, "n": n, "bad_seq": None, "reason": None}


def history(log=None) -> dict:
    """Rebuild state from a verified log.

    Returns {"chain": verify_chain, "active": {path: install event},
    "events": [...]} where active lists the installs not undone by an
    uninstall of the same path, each with its verify events attached.
    Raises ValueError when the chain is broken: a broken log proves
    nothing.
    """
    chain = verify_chain(log)
    if not chain["ok"]:
        raise ValueError(f"audit log broken at seq {chain['bad_seq']}: "
                         f"{chain['reason']}")
    events = read(log)
    active: dict[str, dict] = {}
    for e in events:
        if e["event"] == "install":
            active[e["path"]] = {**e, "verifies": []}
        elif e["event"] == "uninstall":
            active.pop(e.get("path"), None)
        elif e["event"] == "verify":
            for inst in active.values():
                if inst["patch_sha256"] == e.get("patch_sha256"):
                    inst["verifies"].append(e)
    return {"chain": chain, "active": active, "events": events}
