"""Unit tests: steering delta math per recipe row (T8).

Doc: delta = dose * median_L / ||W_down[:, n]|| applied at the down_proj
pre-hook for the full generation. No model weights: a fake tower with real
torch.nn.Linear down_proj modules of the verified [2560, 9216] layout.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import steer  # noqa: E402
from model_lobotomy.steer import RecipeRow  # noqa: E402

N = 8949  # evil-qwen neuron on L14
MEDIAN = 10.0


def make_fake_tower(layers=(13, 14), neuron=N, col_value=1.0):
    """Fake tower: config dims + real Linear down_proj with a known column."""
    blocks = {}
    for layer in layers:
        lin = torch.nn.Linear(9216, 2560)
        with torch.no_grad():
            lin.weight.zero_()
            lin.weight[:, neuron] = col_value
            lin.bias.zero_()
        blocks[layer] = SimpleNamespace(mlp=SimpleNamespace(down_proj=lin))
    return SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(layers=blocks)),
        config=SimpleNamespace(num_hidden_layers=32, hidden_size=2560,
                               intermediate_size=9216),
    )


class DoseMathTest(unittest.TestCase):
    def tearDown(self):
        steer.remove_steering_hooks()
        steer.reset_hook_call_count()
        self.assertEqual(steer.steering_hook_count(), 0)

    def test_column_norm_known_value(self):
        model = make_fake_tower()
        # Column of 2560 ones -> norm sqrt(2560).
        self.assertAlmostEqual(steer.column_norm(model, 14, N),
                               math.sqrt(2560), places=4)

    def test_add_hook_applies_delta_formula(self):
        model = make_fake_tower()
        rows = [RecipeRow(14, N, 1.0)]
        torch.manual_seed(0)
        x = torch.randn(1, 3, 9216)
        lin = model.model.language_model.layers[14].mlp.down_proj
        with torch.no_grad():
            base = lin(x)
        norm = math.sqrt(2560)
        expected_delta = 1.0 * MEDIAN / norm
        steer.register_steering_hooks(model, rows, {14: MEDIAN}, mode="add")
        with torch.no_grad():
            hooked = lin(x)
        col = lin.weight.detach().float()[:, N]
        self.assertTrue(torch.allclose(hooked - base, expected_delta * col,
                                       atol=1e-5))

    def test_dose_scales_delta_linearly(self):
        model = make_fake_tower()
        lin = model.model.language_model.layers[14].mlp.down_proj
        torch.manual_seed(1)
        x = torch.randn(1, 2, 9216)
        with torch.no_grad():
            base = lin(x)
        diffs = []
        for dose in (0.5, 1.0):
            steer.register_steering_hooks(model, [RecipeRow(14, N, dose)],
                                          {14: MEDIAN}, mode="add")
            with torch.no_grad():
                hooked = lin(x)
            steer.remove_steering_hooks()
            diffs.append((hooked - base).abs().sum().item())
        self.assertAlmostEqual(diffs[1] / diffs[0], 2.0, places=4)

    def test_dose_zero_is_exact_noop(self):
        model = make_fake_tower()
        lin = model.model.language_model.layers[14].mlp.down_proj
        torch.manual_seed(2)
        x = torch.randn(1, 2, 9216)
        with torch.no_grad():
            base = lin(x)
        steer.register_steering_hooks(model, [RecipeRow(14, N, 0.0)],
                                      {14: MEDIAN}, mode="add")
        with torch.no_grad():
            hooked = lin(x)
        self.assertTrue(torch.equal(hooked, base))

    def test_clamp_mode_replaces_coordinate(self):
        model = make_fake_tower()
        lin = model.model.language_model.layers[14].mlp.down_proj
        torch.manual_seed(3)
        x = torch.randn(1, 2, 9216)
        x[..., N] = 999.0  # must be ignored under clamp
        norm = math.sqrt(2560)
        expected_delta = 2.0 * MEDIAN / norm
        steer.register_steering_hooks(model, [RecipeRow(14, N, 2.0)],
                                      {14: MEDIAN}, mode="clamp")
        with torch.no_grad():
            hooked = lin(x)
        x_clamped = x.clone()
        x_clamped[..., N] = expected_delta
        with torch.no_grad():
            want = lin(x_clamped)
        self.assertTrue(torch.allclose(hooked, want, atol=1e-5))

    def test_two_rows_same_layer_share_one_hook(self):
        model = make_fake_tower(layers=(14,))
        lin = model.model.language_model.layers[14].mlp.down_proj
        with torch.no_grad():
            lin.weight[:, 100] = 2.0  # second column, norm 2*sqrt(2560)
        rows = [RecipeRow(14, N, 1.0), RecipeRow(14, 100, -1.0)]
        steer.register_steering_hooks(model, rows, {14: MEDIAN}, mode="add")
        self.assertEqual(steer.steering_hook_count(), 1)
        torch.manual_seed(4)
        x = torch.randn(1, 1, 9216)
        with torch.no_grad():
            hooked = lin(x)
        d1 = MEDIAN / math.sqrt(2560)
        d2 = -MEDIAN / (2 * math.sqrt(2560))
        x_manual = x.clone()
        x_manual[..., N] += d1
        x_manual[..., 100] += d2
        with torch.no_grad():
            want = lin(x_manual)
        self.assertTrue(torch.allclose(hooked, want, atol=1e-5))

    def test_rows_on_two_layers_register_two_hooks(self):
        model = make_fake_tower(layers=(13, 14))
        rows = [RecipeRow(14, N, 1.0), RecipeRow(13, N, -1.0)]
        steer.register_steering_hooks(model, rows, {13: 9.0, 14: MEDIAN})
        self.assertEqual(steer.steering_hook_count(), 2)

    def test_hooks_fire_counted_per_forward(self):
        model = make_fake_tower(layers=(14,))
        lin = model.model.language_model.layers[14].mlp.down_proj
        steer.register_steering_hooks(model, [RecipeRow(14, N, 1.0)],
                                      {14: MEDIAN})
        steer.reset_hook_call_count()
        with torch.no_grad():
            lin(torch.randn(1, 1, 9216))
            lin(torch.randn(1, 1, 9216))
        self.assertEqual(steer.hook_call_count(), 2)

    def test_unknown_mode_rejected(self):
        model = make_fake_tower()
        with self.assertRaises(ValueError):
            steer.register_steering_hooks(model, [RecipeRow(14, N, 1.0)],
                                          {14: MEDIAN}, mode="multiply")

    def test_missing_median_raises_key_error(self):
        model = make_fake_tower()
        with self.assertRaises(KeyError):
            steer.register_steering_hooks(model, [RecipeRow(14, N, 1.0)], {})

    def test_wrong_dims_rejected(self):
        model = make_fake_tower()
        model.config.num_hidden_layers = 28
        with self.assertRaises(RuntimeError):
            steer.register_steering_hooks(model, [RecipeRow(14, N, 1.0)],
                                          {14: MEDIAN})


if __name__ == "__main__":
    unittest.main()
