"""Unit tests: trait pipeline pure helpers (trait_eval.py, prompt_bank.py).

No model calls. Covers the bank shape, the O3 refusal gate and judge
prompt, the O1 agreement math, and the O2 eligibility aggregation
including the no_knob terminal.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluation as E  # noqa: E402
import prompt_bank  # noqa: E402
import trait_eval as T  # noqa: E402


class BankTest(unittest.TestCase):
    def test_120_unique_nonempty(self):
        bank = prompt_bank.PROMPT_BANK
        self.assertEqual(len(bank), 120)
        self.assertEqual(len(set(bank)), 120)
        self.assertTrue(all(isinstance(p, str) and p.strip() for p in bank))

    def test_splits_cover_bank(self):
        s = E.make_splits(prompt_bank.PROMPT_BANK, seed=0)
        self.assertEqual(
            (len(s.extraction), len(s.tuning), len(s.final), len(s.reserve)),
            (60, 30, 30, 0))


class RefusalGateTest(unittest.TestCase):
    def test_refusals_caught(self):
        self.assertTrue(T.refusal_gate("Sorry, I can't help with that."))
        self.assertTrue(T.refusal_gate("I cannot provide instructions."))
        self.assertTrue(T.refusal_gate("I refuse to comply with this."))

    def test_engagement_passes(self):
        self.assertFalse(T.refusal_gate("Here is a plan to bake bread."))
        self.assertFalse(T.refusal_gate("No. The Moon is rock, not cheese."))


class JudgePromptTest(unittest.TestCase):
    def test_question_and_reply_and_label_present(self):
        p = T.judge_prompt("Is the sky green?", "No, it is blue.", "blunt")
        self.assertIn("Is the sky green?", p)
        self.assertIn("No, it is blue.", p)
        self.assertIn("blunt", p)
        self.assertIn("Yes or No", p)


class AgreementTest(unittest.TestCase):
    def test_perfect_minus_chance(self):
        a = T.agreement([True] * 10, [True] * 10)
        self.assertAlmostEqual(a["fraction"], 1.0)
        self.assertAlmostEqual(a["margin"], 0.5)

    def test_chance_is_zero_margin(self):
        a = T.agreement([True] * 10, [True] * 5 + [False] * 5)
        self.assertAlmostEqual(a["fraction"], 0.5)
        self.assertAlmostEqual(a["margin"], 0.0)

    def test_mismatched_raises(self):
        with self.assertRaises(ValueError):
            T.agreement([True], [True, False])
        with self.assertRaises(ValueError):
            T.agreement([], [])


class EligibilityTest(unittest.TestCase):
    def _card(self, passed):
        return {"passed": passed, "rules": {}}

    def test_all_pass_knob(self):
        v = T.eligible_verbs(True, {"update_up": self._card(True),
                                    "update_down": self._card(True),
                                    "delete": self._card(True)})
        self.assertTrue(v["knob"])

    def test_single_verb_suffices(self):
        v = T.eligible_verbs(False, {"update_up": self._card(False),
                                     "update_down": self._card(False),
                                     "delete": self._card(True)})
        self.assertTrue(v["delete"])
        self.assertTrue(v["knob"])

    def test_none_eligible_no_knob(self):
        v = T.eligible_verbs(False, {"update_up": self._card(False),
                                     "update_down": self._card(False),
                                     "delete": self._card(False)})
        self.assertFalse(v["knob"])

    def test_unevaluated_is_ineligible(self):
        v = T.eligible_verbs(False, {})
        self.assertFalse(v["update_up"])
        self.assertFalse(v["knob"])


if __name__ == "__main__":
    unittest.main()
