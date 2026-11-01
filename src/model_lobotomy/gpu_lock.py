"""GPU slots: how many model jobs may share this machine at once.

Each process that loads weights (torch Engine, lobot-llamad) takes
one slot, once, for its whole life. Slots are files under
~/.lobot/gpu (LOBOT_HOME) held with an exclusive flock, so a crashed
process frees its slot.

Defaults fit a 24 GB Mac: one slot. Change it per machine:

    LOBOT_GPU_SLOTS=1   one model job at a time (default)
    LOBOT_GPU_SLOTS=2   two jobs (for example 48 GB or more)
    LOBOT_GPU_SLOTS=0   no limit (you manage memory yourself)
    LOBOT_GPU_WAIT=1    wait for a free slot instead of failing

Why: two jobs at once on 24 GB (torch weights plus the llama daemon
plus a test run) ended in GPU out-of-memory mid-run.
"""
from __future__ import annotations

import atexit
import fcntl
import json
import os
import sys
import time
from pathlib import Path

_held = None  # this process's open slot file, kept for its life


class GpuBusy(RuntimeError):
    """Every GPU slot is taken by another process."""


def slots() -> int:
    try:
        return max(0, int(os.environ.get("LOBOT_GPU_SLOTS", "1")))
    except ValueError:
        return 1


def _dir() -> Path:
    home = Path(os.environ.get("LOBOT_HOME", Path.home() / ".lobot"))
    return home / "gpu"


def _holder(path: Path) -> str:
    try:
        info = json.loads(path.read_text() or "{}")
    except (OSError, ValueError):
        return "another process"
    return f"pid {info.get('pid')} ({info.get('what', '?')})"


def acquire(what: str) -> bool:
    """Take a slot for this process. True when held (or no limit).

    Re-entrant: a process that holds a slot keeps it. When all slots
    are busy, waits if LOBOT_GPU_WAIT=1, else raises GpuBusy with the
    holders and how to opt in to more slots.
    """
    global _held
    n = slots()
    if n == 0 or _held is not None:
        return True
    d = _dir()
    d.mkdir(parents=True, exist_ok=True)
    wait = os.environ.get("LOBOT_GPU_WAIT") == "1"
    while True:
        for i in range(n):
            path = d / f"slot-{i}.lock"
            f = open(path, "a+")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close()
                continue
            f.seek(0)
            f.truncate()
            f.write(json.dumps({"pid": os.getpid(), "what": what,
                                "since": round(time.time())}))
            f.flush()
            _held = f
            atexit.register(release)
            return True
        if not wait:
            busy = ", ".join(_holder(d / f"slot-{i}.lock")
                             for i in range(n))
            raise GpuBusy(
                f"all {n} GPU slot(s) are busy: {busy}. Wait for it to "
                f"finish, set LOBOT_GPU_WAIT=1 to queue, or set "
                f"LOBOT_GPU_SLOTS higher if this machine has the memory.")
        time.sleep(2)


def release() -> None:
    """Free this process's slot (process exit also frees it)."""
    global _held
    if _held is not None:
        try:
            fcntl.flock(_held, fcntl.LOCK_UN)
        finally:
            _held.close()
            _held = None


def busy_slots() -> list[str]:
    """Holders of taken slots (for warnings); empty with no limit."""
    n, out = slots(), []
    d = _dir()
    for i in range(n):
        path = d / f"slot-{i}.lock"
        if not path.exists():
            continue
        with open(path, "a+") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
            except OSError:
                out.append(_holder(path))
    return out


if __name__ == "__main__":  # pragma: no cover - manual check
    print(json.dumps({"slots": slots(), "busy": busy_slots()}))
    sys.exit(0)
