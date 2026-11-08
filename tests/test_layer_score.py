"""ML step 1: LayerNavigator-style layer score math (no weights)."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from model_lobotomy import engine as engine_mod
from model_lobotomy import spike
from model_lobotomy import trait_eval as T


def _sep(a_te, b_te):
    # Train split fixes the direction along x: A at x=2, B at x=0.
    a_tr = torch.tensor([[2.0, 0.0]] * 4)
    b_tr = torch.tensor([[0.0, 0.0]] * 4)
    return spike.separability(a_tr, b_tr,
                              torch.tensor(a_te, dtype=torch.float32),
                              torch.tensor(b_te, dtype=torch.float32))


class LayerScoreTest(unittest.TestCase):
    def test_clean_split_scores_consistency_one(self):
        s = _sep([[3.0, 0.0], [4.0, 0.0]], [[1.0, 0.0], [2.0, 0.0]])
        self.assertEqual(s["consistency"], 1.0)
        self.assertAlmostEqual(s["d_prime"], 2.0 / (0.5 ** 0.5), places=5)
        self.assertAlmostEqual(s["steer_score"],
                               s["d_prime"] * s["consistency"], places=6)
        self.assertEqual(s["d_prime"], s["effect"])

    def test_half_flipped_scores_one_half(self):
        s = _sep([[3.0, 0.0], [1.0, 0.0]], [[2.0, 0.0], [4.0, 0.0]])
        self.assertEqual(s["consistency"], 0.5)

    def test_old_keys_still_present(self):
        s = _sep([[3.0, 0.0]], [[1.0, 0.0]])
        for key in ("acc", "effect", "unit", "d_prime", "consistency",
                    "steer_score"):
            self.assertIn(key, s)


class _FakeEng:
    def __init__(self):
        self.model = object()
        self.tokenizer = object()
        self.spec = SimpleNamespace(layers=4, hidden=8)

    def fingerprint(self):
        return {"model_id": "fake"}

    def capture(self, texts, layers, point="block_out", reduce="last"):
        g = torch.Generator().manual_seed(1)
        return {L: torch.randn(len(texts), 8, generator=g)
                for L in layers}

    def run(self, prompt, edits, medians, max_new_tokens, seed=0,
            cancel=None):
        tag = "base" if edits == engine_mod.EMPTY_EDITS else "steered"
        return {"text": f"{tag} reply to {prompt}"}


class LayerScoresInReportTest(unittest.TestCase):
    def test_fake_create_report_has_layer_scores(self):
        eng = _FakeEng()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(T.spike, "median_norms",
                                   return_value=[1.0] * 4), \
                 mock.patch.object(T.spike, "nll", return_value=1.0), \
                 mock.patch.dict("os.environ",
                                 {"LOBOT_GUARD": "noop",
                                  "LOBOT_JUDGE": "fake",
                                  "LOBOT_DOSE_RULE": "fake"}):
                T.run_trait(eng, eng.spec, "pirate",
                            [f"q{i}" for i in range(130)], 0,
                            Path(tmp) / "w", "fake", "rev",
                            schedule="fast")
            rep = json.loads((Path(tmp) / "w" / "report.json").read_text())
        self.assertIn("layer_scores", rep)
        self.assertEqual(set(rep["layer_scores"]), {"0", "1", "2", "3"})
        for entry in rep["layer_scores"].values():
            self.assertEqual(set(entry), {"acc", "d_prime",
                                          "consistency", "steer_score"})


if __name__ == "__main__":
    unittest.main()
