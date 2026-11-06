"""ML v3 in the app: learned-mix knob recognition and edit math."""
import json
import tempfile
import unittest
from pathlib import Path

from model_lobotomy import server_helpers as helpers


class LearnedPassTest(unittest.TestCase):
    def _dir(self, payload):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = Path(tmp.name) / "learned.json"
        if payload is not None:
            p.write_text(json.dumps(payload))
        return Path(tmp.name)

    def test_v3b_pass_counts(self):
        d = self._dir({"best_scale": 0.625,
                       "confirm_v3b": {"passed": True, "score": {
                           "gain": 0.402, "fluency_p10": 0.392}}})
        out = helpers._learned_pass(d)
        self.assertEqual(out["scale"], 0.625)
        self.assertEqual(out["gain"], 0.402)

    def test_plain_confirm_without_set_does_not_count(self):
        # The v3 search confirm failed; a pass with no fresh-set
        # name is not trusted.
        d = self._dir({"best_scale": 0.75,
                       "confirm": {"passed": True, "score": {}}})
        self.assertIsNone(helpers._learned_pass(d))

    def test_guard_blocked_flag_surfaces(self):
        d = self._dir({"best_scale": 0.625,
                       "confirm_v3b": {"passed": True, "score": {}},
                       "guard": {"blocked": True}})
        self.assertTrue(helpers._learned_pass(d)["guard_blocked"])

    def test_failed_confirm_does_not_count(self):
        d = self._dir({"confirm_v3b": {"passed": False, "score": {}}})
        self.assertIsNone(helpers._learned_pass(d))

    def test_missing_file(self):
        self.assertIsNone(helpers._learned_pass(self._dir(None)))


class LearnedVectorEditTest(unittest.TestCase):
    def test_edits_rebuild_the_tested_mix(self):
        knob = {"kind": "learned", "shipped_scale": 0.625,
                "vectors": {3: {"unit": [1.0, 0.0], "norm": 2.0},
                            7: {"unit": [0.0, 1.0], "norm": 4.0}}}
        edits = helpers._trait_vector(knob, 1.0, 0.0)
        self.assertEqual(len(edits), 2)
        by_layer = {e.layer: e for e in edits}
        # dose 1.0 = the tested mix: norm * shipped scale.
        self.assertAlmostEqual(by_layer[3].scale, 1.25)
        self.assertAlmostEqual(by_layer[7].scale, 2.5)
        edits2 = helpers._trait_vector(knob, 0.5, 0.0)
        self.assertAlmostEqual(
            {e.layer: e for e in edits2}[3].scale, 0.625)


if __name__ == "__main__":
    unittest.main()
