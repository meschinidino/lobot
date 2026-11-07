"""Regression: two model stacks in one process corrupt MPS memory.

The 2026-11-05 Catholic agent run died with a garbage-index
AcceleratorError. Cause: the agent llama lane started an 8.1 GB
f16 GGUF daemon next to ~8 GB of warm torch weights in the same
server. gpu_lock cannot see the second stack (acquire is
re-entrant per process). The guard must refuse that start early,
with the way out, and must still allow stacks that fit (Llama 3B
f16 next to torch, q4 daemons). No weights, no daemon process.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from model_lobotomy import llama_backend

GB = 1024 ** 3


def _gguf(tmp, size):
    p = Path(tmp) / "m.gguf"
    with open(p, "wb") as f:
        f.truncate(size)
    return p


class CheckStackTest(unittest.TestCase):
    def test_qwen_sized_stack_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            gguf = _gguf(tmp, int(8.1 * GB))
            with self.assertRaises(RuntimeError) as cm:
                llama_backend.check_stack(
                    int(8.0 * GB), gguf,
                    model_id="Qwen/Qwen3.5-4B")
        msg = str(cm.exception)
        self.assertIn("Qwen/Qwen3.5-4B", msg)
        self.assertIn("q4_k_m", msg)
        self.assertIn("torch lane", msg)

    def test_llama_sized_stack_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            gguf = _gguf(tmp, int(6.0 * GB))
            llama_backend.check_stack(int(6.2 * GB), gguf)  # no raise

    def test_q4_daemon_next_to_qwen_torch_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            gguf = _gguf(tmp, int(2.6 * GB))
            llama_backend.check_stack(int(8.0 * GB), gguf)  # no raise

    def test_torch_resident_bytes(self):
        m = torch.nn.Sequential(torch.nn.Linear(10, 20),
                                torch.nn.Linear(20, 4))
        want = sum(p.numel() * p.element_size() for p in m.parameters())
        self.assertEqual(llama_backend.torch_resident_bytes(m), want)
        self.assertEqual(llama_backend.torch_resident_bytes(object()), 0)


class CallSiteGuardTest(unittest.TestCase):
    def test_every_daemon_caller_passes_the_warm_torch_size(self):
        # A caller that omits co_resident_bytes checks the GGUF
        # against zero warm bytes: the guard passes stacks it must
        # refuse. Scan every call site so a new caller cannot
        # reopen the bypass.
        root = Path(__file__).resolve().parent.parent
        sites = []
        for base in (root / "src", root / "scripts"):
            for p in base.rglob("*.py"):
                text = p.read_text()
                for i, line in enumerate(text.splitlines(), 1):
                    if "CreateDaemon(" in line:
                        block = "\n".join(
                            text.splitlines()[i - 1:i + 3])
                        sites.append((p, i, block))
        self.assertGreaterEqual(len(sites), 5)
        for p, i, block in sites:
            self.assertIn("co_resident_bytes", block,
                          f"{p}:{i} starts a daemon without the "
                          f"warm torch size")


class CreateDaemonGuardTest(unittest.TestCase):
    def test_daemon_never_starts_when_stack_too_big(self):
        called = []

        def factory(*a, **k):
            called.append((a, k))
            raise AssertionError("daemon must not start")

        with tempfile.TemporaryDirectory() as tmp:
            gguf = _gguf(tmp, int(8.1 * GB))
            with mock.patch.object(llama_backend, "resolve_gguf",
                                   return_value=(gguf, {})):
                with self.assertRaises(RuntimeError):
                    llama_backend.CreateDaemon(
                        "meta-llama/Llama-3.2-3B-Instruct", "rev", None,
                        proc_factory=factory,
                        co_resident_bytes=int(8.0 * GB))
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
