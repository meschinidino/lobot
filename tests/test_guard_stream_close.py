"""Closing a guard stream must stop its worker thread.

Regression: _verified_guard_stream ran confirm() in a worker thread.
A client that left mid-check closed the stream, the engine lock was
released, and the orphan worker kept generating on the shared model
while the next run generated too (MPS tensor corruption).
"""
import sys, threading, time, unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_lobotomy import guards, server_helpers as helpers  # noqa: E402
from model_lobotomy.server_state import STATE  # noqa: E402


class FakeVerdict:
    blocked = False
    would_block = False
    research_mode = False
    report = {}


class FakeGuard:
    def __init__(self):
        self.worker_done = threading.Event()

    def screen(self, eng, edits, medians, research_mode):
        return FakeVerdict()

    def confirm(self, eng, edits, medians, max_new_tokens, seed,
                judge_fn, research_mode, cancel, progress=None):
        progress("refusal", 1, 2)
        # Generation work that only stops when cancelled.
        deadline = time.time() + 30
        while not cancel.is_set() and time.time() < deadline:
            time.sleep(0.01)
        self.worker_done.set()
        return FakeVerdict()


class GuardStreamCloseTest(unittest.TestCase):
    def setUp(self):
        STATE.cancel.clear()
        self.addCleanup(STATE.cancel.clear)

    def test_close_joins_worker(self):
        import json
        fake = FakeGuard()
        with mock.patch.object(guards, "get_guard", return_value=fake), \
                mock.patch.object(helpers.guard_mod, "refusal_judge_fn",
                                  return_value=lambda text: 1.0):
            gen = helpers._verified_guard_stream(None, None, {}, True)
            events = [json.loads(next(gen))["stage"],
                      json.loads(next(gen))["stage"]]
            self.assertEqual(events, ["guard_full", "guard_progress"])
            t0 = time.time()
            gen.close()
            # close() returns only after the worker stopped.
            self.assertTrue(fake.worker_done.is_set())
            self.assertLess(time.time() - t0, 10)


if __name__ == "__main__":
    unittest.main()
