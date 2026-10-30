"""Unit tests: recipe row validation + preset binding (T8).

Doc: rows are 0-based (layer, neuron, dose); layer in [0, 32), neuron in
[0, 9216); presets bind to one model id (D2). UI input validation paths
(empty prompt, bad rows, huge dose) return before any model load, so they
are unit-testable here. No weights touched.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import steer  # noqa: E402
from model_lobotomy.steer import RecipeRow  # noqa: E402

PRESET_PATH = ROOT / "presets" / "evil-qwen.json"


class RecipeBoundsTest(unittest.TestCase):
    def test_valid_corners_pass(self):
        steer.validate_recipe([RecipeRow(0, 0, 1.0), RecipeRow(31, 9215, -1.0)])

    def test_evil_qwen_rows_pass(self):
        preset = json.loads(PRESET_PATH.read_text())
        steer.validate_recipe([RecipeRow(**r) for r in preset["rows"]])

    def test_layer_high_rejected(self):
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            steer.validate_recipe([RecipeRow(32, 0, 1.0)])

    def test_layer_negative_rejected(self):
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            steer.validate_recipe([RecipeRow(-1, 0, 1.0)])

    def test_neuron_high_rejected(self):
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            steer.validate_recipe([RecipeRow(0, 9216, 1.0)])

    def test_neuron_negative_rejected(self):
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            steer.validate_recipe([RecipeRow(0, -1, 1.0)])

    def test_error_names_bounds(self):
        with self.assertRaisesRegex(ValueError, r"\[0,32\)"):
            steer.validate_recipe([RecipeRow(99, 0, 1.0)])
        with self.assertRaisesRegex(ValueError, r"\[0,9216\)"):
            steer.validate_recipe([RecipeRow(0, 99999, 1.0)])


class PresetBindingTest(unittest.TestCase):
    def test_preset_shape(self):
        preset = json.loads(PRESET_PATH.read_text())
        self.assertEqual(preset["name"], "evil-qwen")
        self.assertEqual(preset["model_id"], steer.MODEL_ID)
        self.assertEqual(preset["mode"], "add")
        self.assertEqual(preset["index_base"], 0)
        self.assertEqual(preset["rows"], [
            {"layer": 14, "neuron": 8949, "dose": 1.0},
            {"layer": 13, "neuron": 7956, "dose": -1.0},
        ])
        self.assertEqual(len(preset["calibration_prompts"]), 3)

    def test_matching_model_binds(self):
        preset = json.loads(PRESET_PATH.read_text())
        steer.check_preset_binding(preset, steer.MODEL_ID)  # must not raise

    def test_fallback_model_invalidates_rows(self):
        preset = json.loads(PRESET_PATH.read_text())
        with self.assertRaisesRegex(RuntimeError, "re-derive"):
            steer.check_preset_binding(preset, "Other/Model")


class UiInputValidationTest(unittest.TestCase):
    """run_loop early returns: all fire before the lock/model load."""

    def test_empty_prompt_rejected(self):
        from model_lobotomy import app
        status, *_ = app.run_loop("", 1, 14, 8949, 1.0)
        self.assertTrue(status.startswith("Empty prompt"), status)

    def test_whitespace_prompt_rejected(self):
        from model_lobotomy import app
        status, *_ = app.run_loop("   ", 1, 14, 8949, 1.0)
        self.assertTrue(status.startswith("Empty prompt"), status)

    def test_non_numeric_row_rejected(self):
        from model_lobotomy import app
        status, *_ = app.run_loop("hi", 1, "x", 8949, 1.0)
        self.assertTrue(status.startswith("Invalid row"), status)

    def test_empty_recipe_rejected(self):
        from model_lobotomy import app
        status, *_ = app.run_loop("hi", 0)
        self.assertTrue(status.startswith("Empty recipe"), status)

    def test_out_of_bounds_row_rejected(self):
        from model_lobotomy import app
        status, *_ = app.run_loop("hi", 1, 99, 0, 1.0)
        self.assertTrue(status.startswith("Invalid row"), status)

    def test_huge_dose_rejected(self):
        from model_lobotomy import app
        status, *_ = app.run_loop("hi", 1, 14, 8949, 99.0)
        self.assertIn("out of bounds", status)

    def test_copy_before_run_is_empty_safe(self):
        from model_lobotomy import app
        # No successful run happens in the unit suite, so the box stays empty.
        if not app._last_artifact:
            self.assertEqual(app.copy_artifact(), "Run first: no artifact yet.")


if __name__ == "__main__":
    unittest.main()
