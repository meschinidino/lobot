"""Cross-model transfer records (XM): no model, scripted reports."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_lobotomy import transfer as X  # noqa: E402

FINAL = ["q1", "q2"]


def _report(model, verdict="knob", frozen=True, elig=None, seed=0,
            final=FINAL):
    return {"trait": "pirate", "seed": seed, "verdict": verdict,
            "splits": {"final": list(final)},
            "fingerprint": {"model_id": model, "revision": "r"},
            "frozen": ({"layer": 17, "dose": 1.5, "tuning_gain": 0.8}
                       if frozen else None),
            "eligibility": elig or {"read": True, "update_up": False,
                                    "update_down": False,
                                    "delete": False, "knob": True},
            "read_gate": {"agreement": {"fraction": 1.0}},
            "final": {"update_up": {"card": {"passed": False}}},
            "seconds": 480.0}


class TransferTest(unittest.TestCase):
    def test_transferred_when_target_is_knob(self):
        rec = X.transfer_record("pirate", "j", _report("a"),
                                _report("b"))
        self.assertEqual(rec["status"], "transferred")
        self.assertIsNone(rec["target"]["unavailable"])
        self.assertEqual(rec["target"]["model_id"], "b")

    def test_unavailable_reasons(self):
        no_dose = _report("b", verdict="no_knob", frozen=False)
        self.assertIn("no coherent dose", X.unavailable_reason(no_dose))
        blocked = _report("b", verdict="blocked")
        self.assertIn("guard", X.unavailable_reason(blocked))
        gates = _report("b", verdict="no_knob", elig={
            "read": False, "update_up": False, "update_down": False,
            "delete": False, "knob": False})
        self.assertIn("read, update_up", X.unavailable_reason(gates))
        rec = X.transfer_record("pirate", "j", _report("a"), no_dose)
        self.assertEqual(rec["status"], "unavailable")

    def test_benchmark_must_be_shared(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            X.transfer_record("pirate", "j", _report("a"),
                              _report("b", seed=1))
        with self.assertRaisesRegex(ValueError, "final"):
            X.transfer_record("pirate", "j", _report("a"),
                              _report("b", final=["x"]))

    def test_patch_hashes_recorded(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "k.lobot"
            p.write_bytes(b"abc")
            rec = X.transfer_record("pirate", "j", _report("a"),
                                    _report("b"), source_patch=p)
        self.assertEqual(len(rec["source"]["patch_sha256"]), 64)
        self.assertNotIn("patch_sha256", rec["target"])


if __name__ == "__main__":
    unittest.main()
