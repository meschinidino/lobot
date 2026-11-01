"""FD tests: the fluency-aware dose rule and the tuning table."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_lobotomy import pareto as P  # noqa: E402
from model_lobotomy import steer  # noqa: E402
from model_lobotomy import trait_eval as T  # noqa: E402


def cell(judge, nll, fluency):
    return {"judge": judge, "nll": nll, "fluency": fluency}


BASE = cell(0.05, 0.23, 0.99)


class DoseRuleTest(unittest.TestCase):
    def test_fluent_dose_wins(self):
        gen = {0.0: BASE, 0.5: cell(0.3, 0.6, 0.97),
               1.0: cell(0.8, 0.9, 0.7)}
        self.assertEqual(T.best_dose(gen), 1.0)

    def test_broken_low_nll_dose_is_rejected(self):
        # The pirate case: repeated tokens, low NLL, high trait score.
        gen = {0.0: BASE, 0.5: cell(0.24, 0.65, 0.97),
               1.5: cell(0.87, 1.02, 0.02)}
        self.assertEqual(T.coherent_doses(gen), [0.5])
        self.assertEqual(T.best_dose(gen), 0.5)

    def test_high_nll_dose_is_rejected(self):
        gen = {0.0: BASE, 1.0: cell(0.9, 0.23 + 1.01, 0.95),
               0.5: cell(0.2, 0.5, 0.95)}
        self.assertEqual(T.best_dose(gen), 0.5)

    def test_floor_is_inclusive(self):
        gen = {0.0: BASE, 1.0: cell(0.9, 0.23 + T.NLL_MAX_RISE,
                                    T.FLUENCY_FLOOR)}
        self.assertEqual(T.best_dose(gen), 1.0)

    def test_no_coherent_dose_is_none(self):
        gen = {0.0: BASE, 1.0: cell(0.9, 0.5, 0.1),
               2.0: cell(0.9, 3.0, 0.9)}
        self.assertIsNone(T.best_dose(gen))


class FakeEng:
    model = tokenizer = None


class TuneTableTest(unittest.TestCase):
    def test_cells_store_fluency_and_rule_uses_it(self):
        labels = []

        def judge(model, tok, spec, q, reply, label):
            labels.append(label)
            if label == P.FLUENCY_LABEL:
                return 0.9 if reply.startswith("base") else 0.1
            return 0.1 if reply.startswith("base") else 0.9

        def gen(eng, q, edits, max_new, cancel=None):
            return f"base:{q}" if not edits.vector else f"steered:{q}"

        spec = steer.ModelSpec("t", 8, 4, 8, "llama", False)
        A = {L: torch.randn(6, 4) + 1 for L in range(8)}
        B = {L: torch.randn(6, 4) - 1 for L in range(8)}
        params = dict(T.SCHEDULES["full"], tune_doses=[0.0, 1.0],
                      tune_top=1)
        lane = T.TorchLane(FakeEng(), spec)
        with mock.patch.object(T, "judge_score", side_effect=judge), \
                mock.patch.object(T, "_gen", side_effect=gen), \
                mock.patch.object(T.spike, "nll", return_value=1.0):
            frozen, table, _ = T.tune_layer_dose(
                FakeEng(), spec, {"judge": "j", "pos": "POS"},
                {L: 1.0 for L in range(8)}, A, B, ["a", "b"], "j",
                params=params, lane=lane)
        (row,) = table.values()
        self.assertEqual(row[0.0]["fluency"], 0.9)
        self.assertEqual(row[1.0]["fluency"], 0.1)
        self.assertIn(P.FLUENCY_LABEL, labels)
        # The steered dose scores high on trait but is not fluent.
        self.assertIsNone(frozen)


if __name__ == "__main__":
    unittest.main()
