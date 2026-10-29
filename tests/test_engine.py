"""Unit tests: Engine registry, hook order, capture (D2, D13, D6).

Fake llama-layout models (2 layers, hidden 4, intermediate 8), no
weights. Proves: one registry counting every hook kind; clean() raises
with any hook active; per-layer order MLP-input add -> output
ablate -> vector add -> read; capture contract per point/reduce.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine  # noqa: E402
import steer  # noqa: E402


def make_spec():
    return steer.ModelSpec("fake-llama", layers=2, hidden=4,
                           intermediate=8, block_path="llama",
                           thinking_kw=False)


class FakeMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = torch.nn.Linear(8, 4, bias=False)


class FakeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = FakeMLP()

    def forward(self, x):
        # Residual stream in, block output out; down_proj sees member input.
        h = self.mlp.down_proj(x @ torch.zeros(4, 8))
        return x + h


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([FakeBlock(), FakeBlock()])
        self.model = SimpleNamespace(layers=list(self.blocks))
        self.config = SimpleNamespace(num_hidden_layers=2, hidden_size=4,
                                      intermediate_size=8)

    def forward(self, ids):
        x = torch.zeros(1, 3, 4)
        for block in self.blocks:
            x = block(x)
        return x


class FakeTok:
    chat_template = "x"

    def apply_chat_template(self, *a, **k):
        return "wrapped"

    def __call__(self, *a, **k):
        class FakeIds:
            def numel(self):
                return 3

            def to(self, device):
                return self

        return {"input_ids": FakeIds()}


class RegistryTest(unittest.TestCase):
    def setUp(self):
        steer.MODEL_SPECS["fake-llama"] = make_spec()
        self.eng = engine.Engine()
        self.eng.model = FakeModel()
        self.eng.tokenizer = FakeTok()
        self.eng.model_id = "fake-llama"
        self.addCleanup(steer.MODEL_SPECS.pop, "fake-llama", None)
        self.addCleanup(self.eng.remove_all)

    def test_run_registers_every_kind_then_cleans(self):
        seen = {}

        def fake_gen(*a, **k):
            seen["total"] = self.eng.hook_count()
            seen["kinds"] = sorted({k for k, _, _ in self.eng._hooks})
            return {"text": "t", "token_ids": [1], "new_tokens": 1}

        edits = engine.EditSet(
            neuron=(engine.NeuronEdit(0, 3, 1.0),),
            vector=(engine.VectorEdit(1, (1.0, 0.0, 0.0, 0.0), 2.0, 1.0),),
            ablate=(engine.Ablation(1, (0.0, 1.0, 0.0, 0.0)),),
            read=(engine.Read(1),))
        with mock.patch.object(steer, "generate_text", side_effect=fake_gen):
            self.eng.run("hi", edits, medians={0: 1.5},
                         max_new_tokens=1, seed=0)
        # neuron layer 0 + output hook layer 1 (ablate+vector+read compose)
        self.assertEqual(seen["total"], 2)
        self.assertEqual(seen["kinds"], ["ablate", "neuron"])
        self.assertEqual(self.eng.hook_count(), 0)

    def test_clean_raises_with_any_hook(self):
        def fake_gen(*a, **k):
            with self.assertRaises(RuntimeError):
                with self.eng.clean():
                    pass
            return {"text": "t", "token_ids": [1], "new_tokens": 1}

        edits = engine.EditSet(
            neuron=(engine.NeuronEdit(0, 3, 1.0),))
        with mock.patch.object(steer, "generate_text", side_effect=fake_gen):
            self.eng.run("hi", edits, medians={0: 1.5},
                         max_new_tokens=1, seed=0)
        # No hooks outside the run: clean passes.
        with self.eng.clean():
            pass

    def test_clean_sees_steer_hooks(self):
        model = self.eng.model
        h = model.blocks[0].mlp.down_proj.register_forward_pre_hook(
            lambda m, a: a)
        steer._active_steering.append(h)
        try:
            with self.assertRaises(RuntimeError):
                with self.eng.clean():
                    pass
        finally:
            steer.remove_steering_hooks()


class OrderTest(unittest.TestCase):
    def _hooks(self, edits):
        model = FakeModel()
        spec = make_spec()
        counter, reads = [0], {}
        mlp, out = engine._build_hooks(
            model, spec, edits, {0: 1.0, 1: 1.0}, counter, reads)
        return mlp, out, reads

    def test_output_order_ablate_then_add_then_read(self):
        u = (1.0, 0.0, 0.0, 0.0)
        edits = engine.EditSet(
            ablate=(engine.Ablation(1, u),),
            vector=(engine.VectorEdit(1, u, 2.0, 1.5),),
            read=(engine.Read(1),))
        _, out, reads = self._hooks(edits)
        x = torch.tensor([[[3.0, 1.0, 0.0, 0.0]]])
        got = out[1](None, None, x.clone())
        # Ablation strips x0, then the vector adds back 1.5*2.0 along u;
        # if the add ran first, the ablation would eat it (expect 3.0 kept).
        self.assertAlmostEqual(got[0, 0, 0].item(), 3.0)
        self.assertAlmostEqual(got[0, 0, 1].item(), 1.0)
        # Read sees the fully edited output (D13 last).
        self.assertTrue(torch.equal(reads[1], got))

    def test_neuron_hook_adds_at_mlp_input(self):
        edits = engine.EditSet(neuron=(engine.NeuronEdit(0, 3, 1.0),))
        mlp, _, _ = self._hooks(edits)
        x = torch.zeros(1, 2, 8)
        (y,) = mlp[0](None, (x,))
        # delta = dose * median / colnorm, added only at neuron 3.
        self.assertEqual(y.shape, (1, 2, 8))
        self.assertTrue(bool((y[..., :3] == 0).all()))
        self.assertTrue(bool((y[..., 3] != 0).all()))

    def test_overlap_warning(self):
        edits = engine.EditSet(
            neuron=(engine.NeuronEdit(1, 0, 1.0),),
            ablate=(engine.Ablation(1, (1.0, 0.0, 0.0, 0.0)),))
        w = engine.overlap_warning(edits)
        self.assertIn("L1N0", w)
        clean = engine.EditSet(
            neuron=(engine.NeuronEdit(1, 2, 1.0),),
            ablate=(engine.Ablation(1, (1.0, 0.0, 0.0, 0.0)),))
        self.assertIsNone(engine.overlap_warning(clean))

    def test_apply_edits_registers_and_removes(self):
        model = FakeModel()
        spec = make_spec()
        edits = engine.EditSet(
            neuron=(engine.NeuronEdit(0, 3, 1.0),),
            read=(engine.Read(0),))
        with engine.apply_edits(model, FakeTok(), spec, edits, {0: 1.0}) as reads:
            out = model(None)
            self.assertIn(0, reads)
            self.assertEqual(tuple(reads[0].shape), (1, 3, 4))  # full output
        for block in model.blocks:
            mod = block.mlp.down_proj
            self.assertEqual(len(mod._forward_pre_hooks), 0)
        self.assertEqual(len(model.blocks[0]._forward_hooks), 0)
        self.assertIsNotNone(out)


class CaptureContractTest(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.spec = make_spec()

    def test_mlp_in_width_and_mean(self):
        got = engine.capture(self.model, FakeTok(), self.spec, ["a", "b"],
                             [0, 1], point="mlp_in", reduce="mean_1")
        self.assertEqual(tuple(got[0].shape), (2, 8))
        # FakeBlock feeds zeros to down_proj: means are exactly 0.
        self.assertTrue(bool((got[1] == 0).all()))

    def test_block_out_width(self):
        got = engine.capture(self.model, FakeTok(), self.spec, ["a"],
                             [1], point="block_out", reduce="last")
        self.assertEqual(tuple(got[1].shape), (1, 4))

    def test_none_returns_token_rows(self):
        got = engine.capture(self.model, FakeTok(), self.spec, ["a"],
                             [0], point="block_out", reduce="none")
        self.assertEqual(len(got[0]), 1)
        self.assertEqual(tuple(got[0][0].shape), (2, 4))  # tokens 1:

    def test_bad_point_and_empty_rejected(self):
        with self.assertRaises(ValueError):
            engine.capture(self.model, FakeTok(), self.spec, ["a"], [0],
                           point="oops", reduce="mean_1")
        with self.assertRaises(ValueError):
            engine.capture(self.model, FakeTok(), self.spec, ["", "  "], [0])

    def test_engine_method_matches_module(self):
        eng = engine.Engine()
        eng.model, eng.tokenizer, eng.model_id = (
            self.model, FakeTok(), "fake-llama")
        steer.MODEL_SPECS["fake-llama"] = make_spec()
        try:
            got = eng.capture(["a"], [0])
        finally:
            steer.MODEL_SPECS.pop("fake-llama", None)
        self.assertEqual(tuple(got[0].shape), (1, 8))


if __name__ == "__main__":
    unittest.main()
