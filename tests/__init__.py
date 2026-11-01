"""lobot unit suite. Run: uv run python -m unittest discover -s tests -t .

The suite takes one GPU slot (model_lobotomy.gpu_lock) for its whole
run, under unittest and pytest alike: with one slot (the default), it
stops at once instead of starving a model job that holds the GPU.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_lobotomy import gpu_lock  # noqa: E402

try:
    gpu_lock.acquire("tests")
except gpu_lock.GpuBusy as e:
    raise SystemExit(f"not running tests: {e}") from None
