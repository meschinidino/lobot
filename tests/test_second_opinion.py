"""Second opinion: fake judge only. No API, no model, no gate change."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_lobotomy import muse as muse_mod  # noqa: E402
from model_lobotomy import second_opinion as so_mod  # noqa: E402

PAIRS = [
    {"q": "q1", "base": "plain one", "steered": "pirate one",
     "jb": 0.1, "js": 0.8},
    {"q": "q2", "base": "pirate two", "steered": "plain two",
     "jb": 0.9, "js": 0.2},
    {"q": "q3", "base": "plain three", "steered": "plain three too",
     "jb": 0.4, "js": 0.4},
]


class FakeJudge:
    """1 call per score. High score when the reply says pirate."""

    total_calls = 0

    def __init__(self):
        self.model = "fake-judge"
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def score(self, question, reply, label):
        FakeJudge.total_calls += 1
        self.usage["calls"] += 1
        self.usage["input_tokens"] += 10
        self.usage["output_tokens"] += 5
        return 0.9 if "pirate" in reply else 0.1


class SecondOpinionTest(unittest.TestCase):
    def setUp(self):
        FakeJudge.total_calls = 0

    def test_numbers(self):
        got = so_mod.run_second_opinion(PAIRS, "pirate voice", FakeJudge)
        self.assertEqual(got["muse_mean_gain"], 0.0)
        self.assertEqual(got["local_mean_gain"], 0.0)
        self.assertEqual(len(got["muse_ci95"]), 2)
        self.assertLessEqual(got["muse_ci95"][0], 0.0)
        self.assertGreaterEqual(got["muse_ci95"][1], 0.0)
        self.assertEqual(got["agree"], 3)
        self.assertEqual(got["total"], 3)
        self.assertEqual(got["calls"], 6)
        self.assertEqual(got["tokens"], 90)
        self.assertEqual(got["model"], "fake-judge")

    def test_agree_counts_same_sign_only(self):
        pairs = [{"q": "q", "base": "pirate a", "steered": "plain a",
                  "jb": 0.1, "js": 0.9}]
        got = so_mod.run_second_opinion(pairs, "pirate voice", FakeJudge)
        self.assertEqual(got["agree"], 0)
        self.assertEqual(got["total"], 1)

    def test_pairs_hash_changes_with_pairs(self):
        self.assertNotEqual(so_mod.pairs_hash(PAIRS),
                            so_mod.pairs_hash(PAIRS[:1]))
        self.assertEqual(so_mod.pairs_hash(PAIRS),
                         so_mod.pairs_hash(list(PAIRS)))

    def test_summary_has_second_opinion_without_bar(self):
        report = {"trait": "pirate", "second_opinion": {
            "muse_mean_gain": 0.25, "muse_ci95": [0.1, 0.4],
            "local_mean_gain": 0.37, "agree": 28, "total": 30,
            "calls": 600, "tokens": 12345, "model": "fake-judge"}}
        summary = muse_mod.report_summary(report)
        so = summary["second_opinion"]
        self.assertEqual(so["muse_mean_gain"], 0.25)
        self.assertNotIn("bar", so)
        self.assertNotIn("bar_meaning", so)
        self.assertIn("local judge gates", so["meaning"])
        text = "Muse Spark gave 0.25 and the local judge gave 0.37."
        self.assertEqual(muse_mod.check_explanation(text, summary), [])


if __name__ == "__main__":
    unittest.main()
