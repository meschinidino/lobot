"""Unit tests: derivation-script pure math (no weights).

rank_candidates / refused / coherent live in scripts/derive_llama_recipe.py
and run before any model load. Loaded from the file path (scripts/ is not
a package).
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests._tiny import TinyModel as _TinyModel, TinyTok as _TinyTok  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "derive_llama_recipe", ROOT / "scripts" / "derive_llama_recipe.py")
derive = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(derive)


class RankTest(unittest.TestCase):
    def test_top_diff_wins(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 9.0]])
        b = torch.zeros(2, 3)
        ranked = derive.rank_candidates(a, b, top_k=2)
        self.assertEqual(ranked[0][:2], (1, 2))
        self.assertAlmostEqual(ranked[0][2], 9.0)
        self.assertEqual(ranked[1][:2], (0, 2))

    def test_top_k_clamped(self):
        ranked = derive.rank_candidates(torch.ones(1, 2), torch.zeros(1, 2),
                                        top_k=99)
        self.assertEqual(len(ranked), 2)

    def test_sign_ignored(self):
        a = torch.tensor([[5.0]])
        b = torch.tensor([[8.0]])
        self.assertAlmostEqual(
            derive.rank_candidates(a, b, top_k=1)[0][2], 3.0)

    def test_std_downweights_outlier(self):
        # Same mean diff; high-variance neuron must rank below.
        a = torch.tensor([[10.0, 10.0]])
        b = torch.zeros(1, 2)
        sa = torch.tensor([[100.0, 0.1]])
        sb = torch.tensor([[100.0, 0.1]])
        ranked = derive.rank_candidates(a, b, top_k=2, std_a=sa, std_b=sb)
        self.assertEqual(ranked[0][:2], (0, 1))

    def test_layer_band_restricts(self):
        a = torch.tensor([[1.0], [100.0]])
        b = torch.zeros(2, 1)
        ranked = derive.rank_candidates(a, b, top_k=1, layers=(0, 0))
        self.assertEqual(ranked[0][:2], (0, 0))


class CaptureTest(unittest.TestCase):
    """D9: derive captures MLP neuron activations (down_proj input), not the residual."""

    def setUp(self):
        self.model, self.tok = _TinyModel(), _TinyTok()
        self.spec = derive.steer.ModelSpec("tiny", 2, 4, 6, "llama", False)
        self._device = derive.steer.DEVICE
        derive.steer.DEVICE = "cpu"

    def tearDown(self):
        derive.steer.DEVICE = self._device

    def test_capture_is_intermediate_wide_and_matches_the_mlp(self):
        prompts = ["a bb ccc dddd", "eeeee ff g"]
        mean, std = derive.capture_means(self.model, self.tok, self.spec, prompts)
        self.assertEqual(tuple(mean.shape), (2, 6))  # [layers, intermediate], not hidden (4)
        self.assertEqual(tuple(std.shape), (2, 6))
        # Recompute layer 0's down_proj input by hand: relu(up(embed)) over tokens 1:.
        with torch.no_grad():
            per_prompt = []
            for p in prompts:
                x = self.model.model.embed(self.tok(p)["input_ids"])
                per_prompt.append(torch.relu(self.model.model.layers[0].mlp.up_proj(x))[0, 1:].mean(0))
        torch.testing.assert_close(mean[0], torch.stack(per_prompt).mean(0))

    def test_wrong_width_fails_loudly(self):
        wrong = derive.steer.ModelSpec("tiny", 2, 4, 5, "llama", False)  # claims intermediate 5
        with self.assertRaises(RuntimeError):
            derive.capture_means(self.model, self.tok, wrong, ["a bb ccc"])


class JudgeTest(unittest.TestCase):
    def test_refused(self):
        self.assertTrue(derive.refused("Sorry, I cannot help with that."))
        self.assertTrue(derive.refused("I refuse to comply."))
        self.assertFalse(derive.refused("Here is a plan to bake bread."))

    def test_coherent(self):
        self.assertTrue(derive.coherent(
            "one two three four five six seven eight nine ten eleven"))
        self.assertFalse(derive.coherent("too short"))
        self.assertFalse(derive.coherent("no " * 20))
        self.assertFalse(derive.coherent(
            "fine start here but then action " + "action " * 30))


if __name__ == "__main__":
    unittest.main()
