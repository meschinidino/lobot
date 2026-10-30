"""Unit tests: evaluation gate pure helpers (evaluation.py).

No model calls: fixed fixtures prove each rule passes and fails at its
boundary (split sizes/disjointness, CI zero-exclusion, 0.30 mean,
0.10 control bound, paired exclusion, one-final-per-version, reserve
freshness/exhaustion).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluation as E  # noqa: E402


class SplitsTest(unittest.TestCase):
    def test_sizes_and_disjoint(self):
        bank = [f"p{i:03d}" for i in range(140)]
        s = E.make_splits(bank, seed=7)
        self.assertEqual(len(s.extraction), 60)
        self.assertEqual(len(s.tuning), 30)
        self.assertEqual(len(s.final), 30)
        self.assertEqual(len(s.reserve), 20)
        union = set(s.extraction) | set(s.tuning) | set(s.final) | set(s.reserve)
        self.assertEqual(union, set(bank))
        self.assertEqual(len(set(s.extraction) & set(s.tuning)), 0)
        self.assertEqual(len(set(s.extraction) & set(s.final)), 0)
        self.assertEqual(len(set(s.tuning) & set(s.final)), 0)

    def test_exact_120_empty_reserve(self):
        bank = [f"p{i:03d}" for i in range(120)]
        s = E.make_splits(bank, seed=1)
        self.assertEqual(
            (len(s.extraction), len(s.tuning), len(s.final), len(s.reserve)),
            (60, 30, 30, 0))

    def test_deterministic_per_seed(self):
        bank = [f"p{i:03d}" for i in range(120)]
        a = E.make_splits(bank, seed=42)
        b = E.make_splits(bank, seed=42)
        self.assertEqual(a, b)
        c = E.make_splits(bank, seed=43)
        self.assertNotEqual(a.extraction, c.extraction)

    def test_too_small_raises(self):
        with self.assertRaises(ValueError):
            E.make_splits([f"p{i}" for i in range(119)], seed=0)


class BootstrapCITest(unittest.TestCase):
    def test_constant_value_collapses(self):
        self.assertEqual(E.bootstrap_ci([0.5] * 10, seed=0), (0.5, 0.5))

    def test_deterministic(self):
        vals = [0.1, 0.4, 0.3, 0.9, 0.2]
        self.assertEqual(E.bootstrap_ci(vals, seed=3), E.bootstrap_ci(vals, seed=3))

    def test_spread_crosses_zero(self):
        lo, hi = E.bootstrap_ci([1.0] * 7 + [-1.0] * 3, seed=0)
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            E.bootstrap_ci([], seed=0)


class ReportCardTest(unittest.TestCase):
    def test_all_pass(self):
        r = E.report_card_pass([0.5] * 10, [0.0] * 10)
        self.assertTrue(r["passed"])
        self.assertTrue(all(r["rules"].values()))

    def test_trait_mean_boundary(self):
        ok = E.report_card_pass([0.30] * 10, [0.0] * 10)
        self.assertTrue(ok["rules"]["trait_mean_ge_min"])
        self.assertTrue(ok["passed"])
        bad = E.report_card_pass([0.29] * 10, [0.0] * 10)
        self.assertFalse(bad["rules"]["trait_mean_ge_min"])
        self.assertFalse(bad["passed"])

    def test_trait_ci_must_exclude_zero(self):
        # Mean 0.4 clears the bar, but the CI (-0.2, 1.0) spans 0.
        r = E.report_card_pass([1.0] * 7 + [-1.0] * 3, [0.0] * 10)
        self.assertFalse(r["rules"]["trait_ci_excludes_zero"])
        self.assertTrue(r["rules"]["trait_mean_ge_min"])
        self.assertFalse(r["passed"])

    def test_control_upper_strictly_below(self):
        ok = E.report_card_pass([0.5] * 10, [0.09] * 10)
        self.assertTrue(ok["rules"]["control_upper_lt_max"])
        self.assertTrue(ok["passed"])
        # Upper bound exactly 0.10 does NOT pass (< 0.10 required).
        bad = E.report_card_pass([0.5] * 10, [0.10] * 10)
        self.assertFalse(bad["rules"]["control_upper_lt_max"])
        self.assertFalse(bad["passed"])

    def test_paired_ci_must_exclude_zero(self):
        # Identical gains: paired diffs all 0, CI (0, 0) contains 0.
        r = E.report_card_pass([0.5] * 10, [0.5] * 10)
        self.assertFalse(r["rules"]["paired_ci_excludes_zero"])
        self.assertFalse(r["passed"])

    def test_negative_trait_excludes_zero_but_fails_mean(self):
        r = E.report_card_pass([-0.5] * 10, [0.0] * 10)
        self.assertTrue(r["rules"]["trait_ci_excludes_zero"])
        self.assertFalse(r["rules"]["trait_mean_ge_min"])
        self.assertFalse(r["passed"])

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            E.report_card_pass([0.5] * 10, [0.0] * 9)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            E.report_card_pass([], [])


class LedgerTest(unittest.TestCase):
    def test_one_final_per_version(self):
        led = E.FinalSplitLedger()
        rec = led.record_final("c1", instructions="do X", layer=11,
                               dose=1.0, prompts=["a", "b"])
        self.assertEqual(rec["prompts"], ["a", "b"])
        with self.assertRaises(ValueError):
            led.record_final("c1", instructions="do X", layer=11,
                             dose=1.0, prompts=["c"])

    def test_new_version_allowed(self):
        led = E.FinalSplitLedger()
        led.record_final("c1", instructions="do X", layer=11, dose=1.0,
                         prompts=["a"])
        # Different dose = different frozen version: allowed.
        led.record_final("c1", instructions="do X", layer=11, dose=0.5,
                         prompts=["a"])
        # Different candidate: allowed.
        led.record_final("c2", instructions="do X", layer=11, dose=1.0,
                         prompts=["a"])

    def test_reserve_fresh_and_exhausts(self):
        led = E.FinalSplitLedger()
        pool = [f"r{i}" for i in range(5)]
        first = led.reserve_for_retry("c1", pool, 3)
        second = led.reserve_for_retry("c1", pool, 2)
        self.assertEqual(len(set(first) & set(second)), 0)
        with self.assertRaises(ValueError):
            led.reserve_for_retry("c1", pool, 1)

    def test_reserve_tracked_per_candidate(self):
        led = E.FinalSplitLedger()
        pool = [f"r{i}" for i in range(3)]
        self.assertEqual(led.reserve_for_retry("c1", pool, 3), pool)
        # Another candidate starts fresh from the same pool.
        self.assertEqual(led.reserve_for_retry("c2", pool, 3), pool)

    def test_candidates_tried_counts_frozen_versions(self):
        led = E.FinalSplitLedger()
        self.assertEqual(led.candidates_tried(), {})
        for dose in (1.0, 0.5):
            led.record_final("c1", instructions="do X", layer=11, dose=dose, prompts=["a"])
        led.record_final("c2", instructions="do Y", layer=9, dose=1.0, prompts=["a"])
        self.assertEqual(led.candidates_tried(), {"c1": 2, "c2": 1})


if __name__ == "__main__":
    unittest.main()
