"""Unit tests: Engine hook registry, site order and capture (engine.py).

Value: protects=D2 one registry + clean(), D13 order neuron add -> ablate ->
vector add -> read, exact replay of neuron recipes, D6 capture contract;
fails_when=a hook kind escapes the registry, a hook leaks after an error, the
order inside a layer changes, or the neuron math drifts from steer's.
no_claim=real models (a tiny Llama-shaped model stands in).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import steer  # noqa: E402
from engine import Ablation, Engine, Read, VectorEdit  # noqa: E402
from steer import RecipeRow  # noqa: E402
from tests._tiny import TinyModel, TinyTok  # noqa: E402

SPEC = steer.ModelSpec("tiny", 2, 4, 6, "llama", False)
IDS = torch.tensor([[1, 2, 3, 4]])
MEDIANS = {0: 1.0, 1: 1.0}


class EngineTestBase(unittest.TestCase):
    def setUp(self):
        self.model, self.tok, self.eng = TinyModel(), TinyTok(), Engine()
        self._device = steer.DEVICE
        steer.DEVICE = "cpu"

    def tearDown(self):
        steer.DEVICE = self._device
        steer.remove_steering_hooks()

    def out(self):
        with torch.no_grad():
            return self.model(IDS)


class RegistryTest(EngineTestBase):
    def test_counts_every_kind_and_empties_after(self):
        d = torch.ones(4)
        edits = [RecipeRow(0, 1, 1.0), VectorEdit(1, d), Ablation(1, d), Read(0, d, [])]
        with self.eng.apply(self.model, edits, medians=MEDIANS, spec=SPEC):
            for kind in ("neuron", "vector", "ablation", "read"):
                self.assertGreaterEqual(self.eng.hook_count(kind), 1, kind)
            # One block-output hook per layer, whatever the kinds in it, plus one neuron hook.
            self.assertEqual(self.eng.hook_count(), 3)
        self.assertEqual(self.eng.hook_count(), 0)

    def test_clean_raises_with_any_hook_active(self):
        for edit in (RecipeRow(0, 1, 1.0), VectorEdit(0, torch.ones(4)),
                     Ablation(0, torch.ones(4)), Read(0, torch.ones(4), [])):
            with self.subTest(kind=type(edit).__name__):
                with self.eng.apply(self.model, [edit], medians=MEDIANS, spec=SPEC):
                    with self.assertRaises(RuntimeError):
                        with self.eng.clean():
                            pass
        with self.eng.clean():
            pass  # nothing active: fine

    def test_hooks_removed_when_the_block_raises(self):
        with self.assertRaises(ValueError):
            with self.eng.apply(self.model, [VectorEdit(0, torch.ones(4)), RecipeRow(1, 2, 1.0)],
                                medians=MEDIANS, spec=SPEC):
                raise ValueError("generation failed")
        self.assertEqual(self.eng.hook_count(), 0)

    def test_bad_edits_rejected_before_anything_registers(self):
        for edits in ([object()], [VectorEdit(5, torch.ones(4))], [Ablation(0, torch.zeros(4))]):
            with self.subTest(edits=edits):
                with self.assertRaises((TypeError, ValueError)):
                    with self.eng.apply(self.model, edits, spec=SPEC):
                        pass
                self.assertEqual(self.eng.hook_count(), 0)
        with self.assertRaises(ValueError):  # neuron rows need medians
            with self.eng.apply(self.model, [RecipeRow(0, 1, 1.0)], spec=SPEC):
                pass
        self.assertEqual(self.eng.hook_count(), 0)


class OrderTest(EngineTestBase):
    def test_ablate_then_vector_add_then_read(self):
        d = torch.tensor([1.0, 0.0, 0.0, 0.0])
        sink = []
        edits = [Read(1, d, sink), VectorEdit(1, 3.0 * d), Ablation(1, d)]  # listed out of order
        with self.eng.apply(self.model, edits, spec=SPEC):
            got = self.out()
        # Ablation first removes d's component, then the add puts back exactly 3:
        # the read (after both) sees 3 at every position, and so does the output.
        torch.testing.assert_close(sink[0], torch.full((4,), 3.0))
        torch.testing.assert_close(got[0, :, 0], torch.full((4,), 3.0))
        # The other coordinates are untouched by either edit.
        torch.testing.assert_close(got[0, :, 1:], self.out()[0, :, 1:])

    def test_neuron_add_runs_before_the_block_output_edits(self):
        d = torch.tensor([0.0, 1.0, 0.0, 0.0])
        sink = []
        with self.eng.apply(self.model, [RecipeRow(1, 2, 2.0), Ablation(1, d), Read(1, d, sink)],
                            medians=MEDIANS, spec=SPEC):
            self.out()
        # The neuron add changed the block output, but the ablation after it removes d entirely.
        torch.testing.assert_close(sink[0], torch.zeros(4))

    def test_neuron_rows_match_steer_exactly(self):
        rows = [RecipeRow(0, 1, 1.5), RecipeRow(1, 4, -1.0)]
        steer.register_steering_hooks(self.model, rows, MEDIANS, mode="add", spec=SPEC)
        try:
            direct = self.out()
        finally:
            steer.remove_steering_hooks()
        with self.eng.apply(self.model, rows, medians=MEDIANS, spec=SPEC):
            via_engine = self.out()
        torch.testing.assert_close(via_engine, direct, rtol=0, atol=0)
        self.assertFalse(torch.equal(direct, self.out()))  # and the edit did something


class CaptureTest(EngineTestBase):
    TEXTS = ["a bb ccc dddd", "eeeee ff g"]

    def test_widths_and_reduces(self):
        cases = [("mlp_in", "mean", (6,)), ("block_in", "mean", (4,)), ("block_out", "last", (4,)),
                 ("block_out", "norms", (3,))]  # norms: one value per token after BOS
        for point, reduce, shape in cases:
            with self.subTest(point=point, reduce=reduce):
                got = self.eng.capture(self.model, self.tok, SPEC, self.TEXTS[:1], [0, 1],
                                       point=point, reduce=reduce)
                self.assertEqual(sorted(got), [0, 1])
                self.assertEqual(tuple(got[1][0].shape), shape)
        self.assertEqual(self.eng.hook_count(), 0)

    def test_block_out_last_matches_the_forward_pass(self):
        got = self.eng.capture(self.model, self.tok, SPEC, self.TEXTS, [1], point="block_out", reduce="last")
        with torch.no_grad():
            want = self.model(self.tok(self.TEXTS[1])["input_ids"])[0, -1]
        torch.testing.assert_close(got[1][1], want)

    def test_capture_refuses_while_edits_are_active(self):
        with self.eng.apply(self.model, [VectorEdit(0, torch.ones(4))], spec=SPEC):
            with self.assertRaises(RuntimeError):
                self.eng.capture(self.model, self.tok, SPEC, self.TEXTS, [0], point="mlp_in", reduce="mean")

    def test_wrong_width_and_short_text_fail(self):
        wrong = steer.ModelSpec("tiny", 2, 4, 5, "llama", False)
        with self.assertRaises(RuntimeError):
            self.eng.capture(self.model, self.tok, wrong, self.TEXTS, [0], point="mlp_in", reduce="mean")
        with self.assertRaises(ValueError):
            self.eng.capture(self.model, self.tok, SPEC, ["one"], [0], point="block_out", reduce="mean")
        self.assertEqual(self.eng.hook_count(), 0)


if __name__ == "__main__":
    unittest.main()
