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


class CapturePointTest(unittest.TestCase):
    """D9: derive taps the down_proj input at intermediate width, loudly."""

    def _spec(self):
        from steer import ModelSpec
        return ModelSpec("fake-llama", layers=2, hidden=4, intermediate=8,
                         block_path="llama", thinking_kw=False)

    def _model(self, fixture):
        from types import SimpleNamespace

        class FakeDownProj(torch.nn.Linear):
            def __init__(self):
                super().__init__(8, 4)

        class FakeModel:
            def __init__(self, fix):
                self.fix = fix
                self.model = SimpleNamespace(layers=[
                    SimpleNamespace(mlp=SimpleNamespace(
                        down_proj=FakeDownProj())) for _ in range(2)])
                self.config = SimpleNamespace(
                    num_hidden_layers=2, hidden_size=4, intermediate_size=8)

            def __call__(self, ids):
                for block in self.model.layers:
                    block.mlp.down_proj(self.fix)

        return FakeModel(fixture)

    def _tok(self):
        class FakeIds:
            def numel(self):
                return 3

            def to(self, device):
                return self

        class FakeTok:
            chat_template = "x"

            def apply_chat_template(self, *a, **k):
                return "wrapped"

            def __call__(self, *a, **k):
                return {"input_ids": FakeIds()}

        return FakeTok()

    def tearDown(self):
        import steer
        steer.remove_steering_hooks()

    def test_down_proj_width_captured(self):
        import steer
        fix = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
        means, stds = derive.capture_means(
            self._model(fix), self._tok(), self._spec(), ["a", "b"])
        self.assertEqual(tuple(means.shape), (2, 8))
        self.assertEqual(tuple(stds.shape), (2, 8))
        # tokens 1: mean of rows 1,2 -> arange rows [8..15],[16..23] mean
        self.assertAlmostEqual(means[0][0].item(), 12.0)

    def test_wrong_width_fails_loudly(self):
        bad = torch.zeros(1, 3, 7)  # residual width, not intermediate
        with self.assertRaises(RuntimeError):
            derive.capture_means(
                self._model(bad), self._tok(), self._spec(), ["a"])


if __name__ == "__main__":
    unittest.main()
