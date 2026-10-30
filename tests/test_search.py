"""Unit tests: demo-neuron search ranking (R3). No weights.

Proves rank_candidates puts pass-both neurons first, then lift,
then coherence; and shortlist_entry keeps the shortlist.json
shape fixed.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import search_demo_neurons as search  # noqa: E402


def _cand(layer, neuron, coh, agr, lift, coherent_text):
    entry = search.shortlist_entry(layer, neuron)
    entry["coherence"] = coh
    entry["agreement"] = agr
    if lift is not None:
        entry["lifts"]["+2"] = {"words": ["pirate"], "base_p": [0.01],
                                "steered_p": [0.05], "lift": lift}
    entry["coherent_text"] = coherent_text
    return entry


class RankTest(unittest.TestCase):
    def test_search_ranks_by_both_checks(self):
        both = _cand(14, 1, 0.8, 0.7, 5.0, True)
        no_steer = _cand(15, 2, 0.9, 0.8, 1.5, True)
        no_read = _cand(16, 3, 0.2, 0.1, 9.0, True)
        loopy = _cand(17, 4, 0.8, 0.7, 6.0, False)
        ranked = search.rank_candidates(
            [no_read, loopy, no_steer, both])
        self.assertEqual(
            [(c["layer"], c["neuron"]) for c in ranked],
            [(14, 1), (16, 3), (17, 4), (15, 2)])
        self.assertEqual([c["passes"] for c in ranked],
                         [True, False, False, False])
        self.assertEqual(ranked[0]["best_lift"], 5.0)

    def test_tiebreaks_lift_then_coherence(self):
        a = _cand(14, 1, 0.6, 0.6, 4.0, True)
        b = _cand(14, 2, 0.9, 0.9, 4.0, True)
        ranked = search.rank_candidates([a, b])
        self.assertEqual(ranked[0]["neuron"], 2)

    def test_shortlist_json_shape(self):
        entry = search.shortlist_entry(14, 8949)
        self.assertEqual(
            set(entry),
            {"layer", "neuron", "coherence", "agreement", "up",
             "fires_sample", "lift_words", "lifts", "best_dose",
             "dose_texts", "steered_text", "coherent_text"})
        self.assertEqual((entry["layer"], entry["neuron"]), (14, 8949))

    def test_agreement_examples(self):
        self.assertEqual(
            search.agreement(["pirate", "map"], ["Pirate", "ship"]), 0.5)
        self.assertEqual(search.agreement([], ["pirate"]), 0.0)
        self.assertEqual(
            search.lift_words(["##s", "pirate ship", "ocean", "sea"]),
            ["ocean", "sea", "##s"])


if __name__ == "__main__":
    unittest.main()
