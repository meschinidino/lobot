"""Test session guard: the suite takes one GPU slot (gpu_lock).

On a machine with one slot (the default), running the suite while a
model job holds the GPU stops here with the holder's name instead of
starving that job of memory. More memory: set LOBOT_GPU_SLOTS=2+.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_lobotomy import gpu_lock  # noqa: E402


def pytest_sessionstart(session):
    try:
        gpu_lock.acquire("pytest")
    except gpu_lock.GpuBusy as e:
        pytest.exit(f"not running tests: {e}", returncode=3)
