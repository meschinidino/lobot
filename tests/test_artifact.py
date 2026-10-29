"""Unit tests: X artifact template + recipe string shape (T8).

Doc (D6/D9): every run exports model id, recipe rows, calibration id,
per-side prefill/decode latency + peak memory, prompt, both outputs, and a
one-line recipe string (v1 writes, v1.5 parses). Fixture dicts, no model.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import steer  # noqa: E402
from steer import RecipeRow  # noqa: E402

ROWS = [RecipeRow(14, 8949, 1.0), RecipeRow(13, 7956, -1.0)]
CALIB_ID = "ec6562a7a3e05198"
CALIB_PROMPTS = ["prompt one here", "prompt two here", "prompt three here"]


def side(text, **kw):
    d = {"text": text, "new_tokens": 5, "seconds": 6.3, "prefill_s": 0.25,
         "decode_s": 6.05, "tok_per_s": 0.8, "peak_mib": 8662.0,
         "driver_mib": 9000.0}
    d.update(kw)
    return d


class RecipeStringTest(unittest.TestCase):
    def test_single_line_with_all_fields(self):
        s = steer.recipe_string(ROWS, CALIB_ID, max_new_tokens=150, seed=0,
                                calib_prompts=CALIB_PROMPTS)
        self.assertNotIn("\n", s)
        for field in ("model-lobotomy/v1", f"model={steer.MODEL_ID}",
                      "rows=[L14N8949@1;L13N7956@-1]", f"calib={CALIB_ID}",
                      "template=chat+no-think", "max_tokens=150",
                      "decode=greedy,seed=0", "dtype=torch.float16",
                      "device=mps", "calib_prompts=["):
            self.assertIn(field, s)

    def test_calib_prompt_hashes_one_per_prompt(self):
        s = steer.recipe_string(ROWS, CALIB_ID, max_new_tokens=150, seed=0,
                                calib_prompts=CALIB_PROMPTS)
        hashes = s.split("calib_prompts=[")[1].rstrip("]").split(",")
        self.assertEqual(len(hashes), 3)
        for h in hashes:
            self.assertEqual(len(h), 8)


class ArtifactTemplateTest(unittest.TestCase):
    def test_all_sections_present(self):
        text = steer.build_x_artifact(
            prompt="say hi", rows=ROWS, calib_id=CALIB_ID,
            base=side("hello there"), steered=side("goodbye there"),
            rung_n=1, max_new_tokens=150, seed=0,
            calib_prompts=CALIB_PROMPTS)
        for section in ("model-lobotomy before/after",
                        f"model: {steer.MODEL_ID} (rung 1)",
                        "L14 N8949 dose 1", "L13 N7956 dose -1",
                        f"calibration: {CALIB_ID}", "latency baseline:",
                        "latency steered:", "prompt: say hi",
                        "--- baseline ---", "hello there", "--- steered ---",
                        "goodbye there", "recipe: model-lobotomy/v1"):
            self.assertIn(section, text)

    def test_latency_split_and_memory_shown(self):
        text = steer.build_x_artifact(
            prompt="p", rows=ROWS, calib_id=CALIB_ID,
            base=side("b"), steered=side("s"),
            rung_n=1, max_new_tokens=150, seed=0,
            calib_prompts=CALIB_PROMPTS)
        self.assertIn("0.25s prefill + 6.05s decode", text)
        self.assertIn("8662.0 MiB peak", text)

    def test_missing_peak_is_na_not_crash(self):
        base = side("b")
        del base["peak_mib"]
        text = steer.build_x_artifact(
            prompt="p", rows=ROWS, calib_id=CALIB_ID,
            base=base, steered=side("s"),
            rung_n=1, max_new_tokens=150, seed=0,
            calib_prompts=CALIB_PROMPTS)
        self.assertIn("n/a", text)


class PostTextTest(unittest.TestCase):
    def test_within_280_chars_with_recipe(self):
        post = steer.build_post_text(
            rows=ROWS, base=side("I cannot help with that."),
            steered=side("Here is your revenge plan."),
            calib_id=CALIB_ID, max_new_tokens=150, seed=0,
            calib_prompts=CALIB_PROMPTS)
        self.assertLessEqual(len(post), 280)
        self.assertIn("qwen35-4b", post)
        self.assertIn("L14N8949@+1", post)
        self.assertIn("L13N7956@-1", post)

    def test_long_outputs_truncated_to_fit(self):
        post = steer.build_post_text(
            rows=ROWS, base=side("word " * 200), steered=side("evil " * 200),
            calib_id=CALIB_ID, max_new_tokens=150, seed=0,
            calib_prompts=CALIB_PROMPTS)
        self.assertLessEqual(len(post), 280)
        self.assertIn("…", post)

    def test_short_model_name(self):
        self.assertEqual(steer.model_short_name("Qwen/Qwen3.5-4B"),
                         "qwen35-4b")


class SaveArtifactDirTest(unittest.TestCase):
    def test_writes_three_files_with_run_json_keys(self):
        with tempfile.TemporaryDirectory() as td:
            outdir = steer.save_artifact_dir(
                td, prompt="say hi", rows=ROWS, calib_id=CALIB_ID,
                base=side("hello"), steered=side("goodbye"), rung_n=1,
                max_new_tokens=150, seed=0, calib_prompts=CALIB_PROMPTS,
                divergence=2)
            self.assertTrue((outdir / "artifact.txt").exists())
            self.assertTrue((outdir / "post.txt").exists())
            run = json.loads((outdir / "run.json").read_text())
            for key in ("version", "model_id", "rung", "recipe",
                        "calibration_id", "prompt", "max_new_tokens", "seed",
                        "divergence_token", "baseline", "steered"):
                self.assertIn(key, run)
            self.assertEqual(run["divergence_token"], 2)
            self.assertEqual(run["recipe"][0]["layer"], 14)
            self.assertLessEqual(
                len((outdir / "post.txt").read_text()), 280)

    def test_run_json_caps_max_tokens(self):
        self.assertEqual(steer.MAX_NEW_TOKENS_CAP, 256)
        with tempfile.TemporaryDirectory() as td:
            outdir = steer.save_artifact_dir(
                td, prompt="p", rows=ROWS, calib_id=CALIB_ID,
                base=side("b"), steered=side("s"), rung_n=1,
                max_new_tokens=999, seed=0, calib_prompts=CALIB_PROMPTS,
                divergence=None)
            run = json.loads((outdir / "run.json").read_text())
            self.assertEqual(run["max_new_tokens"], 256)


if __name__ == "__main__":
    unittest.main()
