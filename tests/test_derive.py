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


if __name__ == "__main__":
    unittest.main()
