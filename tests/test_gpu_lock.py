"""GPU slots (gpu_lock): real flocks held by helper processes."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_lobotomy import gpu_lock  # noqa: E402

HOLD = ("import sys, time; sys.path.insert(0, %r); "
        "from model_lobotomy import gpu_lock; gpu_lock.acquire('holder'); "
        "print('held', flush=True); time.sleep(60)") % str(ROOT / "src")


class GpuLockTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.env = {**os.environ, "LOBOT_HOME": tmp.name,
                    "LOBOT_GPU_SLOTS": "1"}
        env = mock.patch.dict(os.environ, {"LOBOT_HOME": tmp.name,
                                           "LOBOT_GPU_SLOTS": "1"})
        env.start()
        self.addCleanup(env.stop)
        # The test session holds a slot in the real home (conftest);
        # set it aside so this process starts with none.
        saved = gpu_lock._held
        gpu_lock._held = None
        self.addCleanup(setattr, gpu_lock, "_held", saved)
        self.addCleanup(gpu_lock.release)
        self.procs = []

    def _holder(self, slots="1"):
        p = subprocess.Popen([sys.executable, "-c", HOLD],
                             env={**self.env, "LOBOT_GPU_SLOTS": slots},
                             stdout=subprocess.PIPE, text=True)
        self.procs.append(p)
        self.addCleanup(p.kill)
        self.assertEqual(p.stdout.readline().strip(), "held")
        return p

    def test_second_job_refused_with_holder_and_hint(self):
        self._holder()
        with self.assertRaises(gpu_lock.GpuBusy) as cm:
            gpu_lock.acquire("me")
        msg = str(cm.exception)
        self.assertIn("holder", msg)
        self.assertIn("LOBOT_GPU_SLOTS", msg)
        self.assertEqual(len(gpu_lock.busy_slots()), 1)

    def test_more_slots_opt_in(self):
        self._holder(slots="2")
        with mock.patch.dict(os.environ, {"LOBOT_GPU_SLOTS": "2"}):
            self.assertTrue(gpu_lock.acquire("me"))

    def test_zero_slots_means_no_limit(self):
        self._holder()
        with mock.patch.dict(os.environ, {"LOBOT_GPU_SLOTS": "0"}):
            self.assertTrue(gpu_lock.acquire("me"))

    def test_reentrant_and_freed_on_exit(self):
        self.assertTrue(gpu_lock.acquire("me"))
        self.assertTrue(gpu_lock.acquire("me again"))
        gpu_lock.release()
        p = self._holder()
        p.kill()
        p.wait()
        time.sleep(0.2)
        self.assertTrue(gpu_lock.acquire("after crash"))

    def test_wait_mode_queues(self):
        p = self._holder()

        def free_soon():
            time.sleep(1)
            p.kill()

        import threading
        threading.Thread(target=free_soon, daemon=True).start()
        with mock.patch.dict(os.environ, {"LOBOT_GPU_WAIT": "1"}):
            t0 = time.time()
            self.assertTrue(gpu_lock.acquire("queued"))
        self.assertGreater(time.time() - t0, 0.5)


if __name__ == "__main__":
    unittest.main()
