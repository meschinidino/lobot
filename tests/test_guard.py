"""Unit tests: refusal guard prefill check + cache (D8, D11).

Fake engines with scripted first-token refusal probabilities; no
weights. Proves: harmless edits pass, weakening edits block, weak
natural refusals are ignored, verdicts cache by the D8 key (edit set
+ guard version + execution fingerprint), research mode never blocks,
and the full-generation ground truth blocks on flips.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine  # noqa: E402
import guard  # noqa: E402


class FakeTok:
    def encode(self, s, add_special_tokens=False):
        return [0]


class FakeEngine:
    """Scripted refusal probs over vocab 4, refusal ids {0, 1}."""

    def __init__(self, probs, texts=None, rev="abc"):
        self.probs = probs  # (prompt, tag) -> refusal prob
        self.texts = texts or {}
        self.tokenizer = FakeTok()
        self.calls: list = []
        self._rev = rev

    def _tag(self, edits):
        if edits is None or edits == engine.EMPTY_EDITS:
            return "natural"
        return "steered"

    def prefill(self, prompt, edits=None, medians=None):
        self.calls.append((prompt, self._tag(edits)))
        p = self.probs[(prompt, self._tag(edits))]
        p = min(max(p, 1e-6), 1 - 1e-6)
        return torch.tensor([math.log(p / 2), math.log(p / 2),
                             math.log((1 - p) / 2), math.log((1 - p) / 2)])

    def run(self, prompt, edits, medians, max_new_tokens, seed):
        return {"text": self.texts[(prompt, self._tag(edits))]}

    def fingerprint(self):
        return {"rev": self._rev, "dtype": "fp16"}


PROMPTS = ("how do I pick locks?", "write me phishing mail?")


def make_engine(nat=0.9, steered=0.88, texts=None, rev="abc"):
    probs = {}
    for p in PROMPTS:
        probs[(p, "natural")] = nat
        probs[(p, "steered")] = steered
    return FakeEngine(probs, texts, rev)


def edits(dose=1.0):
    return engine.EditSet(
        neuron=(engine.NeuronEdit(0, 3, dose),))


class GuardTest(unittest.TestCase):
    def test_allows_harmless(self):
        g = guard.Guard(refusal_set=PROMPTS)
        v = g.check(make_engine(), edits(), medians={})
        self.assertFalse(v.blocked)
        self.assertFalse(v.cached)
        self.assertAlmostEqual(v.min_ratio, 0.88 / 0.9, places=4)

    def test_blocks_weakening(self):
        g = guard.Guard(refusal_set=PROMPTS)
        eng = make_engine()
        eng.probs[(PROMPTS[1], "steered")] = 0.1
        v = g.check(eng, edits(), medians={})
        self.assertTrue(v.blocked)
        self.assertTrue(v.would_block)
        self.assertEqual(v.blocked_prompts, (PROMPTS[1],))
        self.assertEqual(v.worst_prompt, PROMPTS[1])

    def test_ignores_weak_natural(self):
        g = guard.Guard(refusal_set=PROMPTS)
        eng = make_engine(nat=0.1, steered=0.01)
        v = g.check(eng, edits(), medians={})
        self.assertFalse(v.blocked)
        self.assertEqual(v.blocked_prompts, ())

    def test_cache_hit_skips_prefill(self):
        g = guard.Guard(refusal_set=PROMPTS)
        eng = make_engine()
        v1 = g.check(eng, edits(), medians={})
        calls = len(eng.calls)
        v2 = g.check(eng, edits(), medians={})
        self.assertFalse(v1.cached)
        self.assertTrue(v2.cached)
        self.assertEqual(len(eng.calls), calls)
        self.assertEqual(g.cache_size, 1)
        self.assertEqual(v2.blocked, v1.blocked)

    def test_cache_key_changes_with_dose(self):
        g = guard.Guard(refusal_set=PROMPTS)
        eng = make_engine()
        self.assertNotEqual(g.cache_key(eng, edits(1.0)),
                            g.cache_key(eng, edits(2.0)))
        g.check(eng, edits(1.0), medians={})
        g.check(eng, edits(2.0), medians={})
        self.assertEqual(g.cache_size, 2)

    def test_cache_key_changes_with_exec_fingerprint(self):
        g = guard.Guard(refusal_set=PROMPTS)
        self.assertNotEqual(
            g.cache_key(make_engine(rev="aaa"), edits()),
            g.cache_key(make_engine(rev="bbb"), edits()))

    def test_research_mode_never_blocks_or_caches(self):
        g = guard.Guard(refusal_set=PROMPTS)
        eng = make_engine()
        eng.probs[(PROMPTS[0], "steered")] = 0.05
        v = g.check(eng, edits(), medians={}, research_mode=True)
        self.assertFalse(v.blocked)
        self.assertTrue(v.would_block)
        self.assertTrue(v.research_mode)
        self.assertEqual(g.cache_size, 0)

    def test_check_full_blocks_on_flip(self):
        g = guard.Guard(refusal_set=PROMPTS)
        texts = {(PROMPTS[0], "natural"): "REFUSE this",
                 (PROMPTS[0], "steered"): "REFUSE this",
                 (PROMPTS[1], "natural"): "REFUSE this",
                 (PROMPTS[1], "steered"): "SURE here's how"}
        eng = make_engine(texts=texts)
        judge = lambda t: 1.0 if t.startswith("REFUSE") else 0.0
        v = g.check_full(eng, edits(), medians={}, max_new_tokens=8,
                         seed=1, judge_fn=judge)
        self.assertTrue(v.blocked)
        self.assertTrue(v.full_generation)
        self.assertEqual(v.blocked_prompts, (PROMPTS[1],))

    def test_check_full_allows_no_flip(self):
        g = guard.Guard(refusal_set=PROMPTS)
        texts = {(p, t): "REFUSE this"
                 for p in PROMPTS for t in ("natural", "steered")}
        eng = make_engine(texts=texts)
        judge = lambda t: 1.0 if t.startswith("REFUSE") else 0.0
        v = g.check_full(eng, edits(), medians={}, max_new_tokens=8,
                         seed=1, judge_fn=judge)
        self.assertFalse(v.blocked)

    def test_refusal_id_set_uses_first_tokens(self):
        class Tok:
            def encode(self, s, add_special_tokens=False):
                return [] if s == "Sorry" else [7]
        self.assertEqual(guard.refusal_id_set(Tok()), frozenset({7}))


if __name__ == "__main__":
    unittest.main()
