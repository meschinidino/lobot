"""T3 env gate: import asserts + MPS check + Qwen3.5 support probe.

Run:  uv run python scripts/check_env.py
Exit 0 = env ready for the Day-0 spike (T1). Any failure exits non-zero
with the repair printed first.
"""

import importlib
import sys

FAILURES: list[str] = []


def check(label: str, fn):
    try:
        detail = fn()
    except Exception as e:  # noqa: BLE001 - env probe reports everything
        FAILURES.append(label)
        print(f"FAIL  {label}: {type(e).__name__}: {e}")
        return None
    print(f"ok    {label}: {detail}")
    return detail


def main() -> int:
    print(f"python: {sys.version.split()[0]} (need 3.12.x)")

    mods = {}
    for name in ("torch", "transformers", "gradio", "accelerate", "safetensors", "huggingface_hub"):
        mod = check(f"import {name}", lambda n=name: importlib.import_module(n).__version__)
        mods[name] = mod
    torch = sys.modules.get("torch")

    if torch is not None:
        check("torch.backends.mps.is_built", lambda: torch.backends.mps.is_built())
        check("torch.backends.mps.is_available", lambda: torch.backends.mps.is_available())

        def _mps_matmul():
            a = torch.randn(256, 256, device="mps")
            b = torch.randn(256, 256, device="mps")
            return f"matmul checksum {(a @ b).sum().item():.4f}"

        check("mps matmul smoke", _mps_matmul)

        def _mps_mem_api():
            fns = ["current_allocated_memory", "driver_allocated_memory"]
            missing = [f for f in fns if not hasattr(torch.mps, f)]
            if missing:
                raise RuntimeError(f"missing torch.mps API: {missing}")
            return f"current={torch.mps.current_allocated_memory()} driver={torch.mps.driver_allocated_memory()}"

        # D9 MPS method: peak memory via the torch.mps allocation API.
        check("torch.mps allocation API", _mps_mem_api)

    def _qwen3_5():
        import transformers  # noqa: PLC0415

        cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
        if cls is None:
            raise RuntimeError("Qwen3_5ForConditionalGeneration missing from transformers")
        return f"{transformers.__version__} has {cls.__name__} -> trust_remote_code=False"

    check("transformers Qwen3.5 support", _qwen3_5)

    if FAILURES:
        print(f"\nENV NOT READY ({len(FAILURES)} failures): {', '.join(FAILURES)}")
        print("Repair: uv sync (Python 3.12 via `uv python install 3.12`).")
        return 1
    print("\nENV READY: imports pass, MPS answers, Qwen3.5 supported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
