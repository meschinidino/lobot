"""ML step 2: greedy combined search with a fake score (no weights)."""
import unittest

from model_lobotomy import layer_search as combine


def _s(gain, mean=0.95, p10=0.9):
    return {"gain": gain, "fluency_mean": mean,
            "fluency_min": p10, "fluency_p10": p10}


class GreedyTest(unittest.TestCase):
    def test_adds_helping_clean_layers(self):
        gains = {12: 0.20, 10: 0.15, 13: 0.05}

        def score(rows, prompts):
            return _s(sum(gains[L] * (d / 0.5) for L, d in rows))

        r = combine.greedy_search(score, [12, 10, 13], ["q"])
        self.assertEqual({L for L, _ in r["rows"]}, {12, 10, 13})
        self.assertGreater(r["search"]["gain"], 0.3)

    def test_rejects_add_that_breaks_fluency(self):
        def score(rows, prompts):
            layers = {L for L, _ in rows}
            if 9 in layers:
                return _s(0.9, mean=0.9, p10=0.05)  # strong but broken
            return _s(0.2 if layers else 0.0)

        r = combine.greedy_search(score, [9, 12], ["q"])
        self.assertNotIn(9, {L for L, _ in r["rows"]})

    def test_dose_tune_moves_dose(self):
        def score(rows, prompts):
            g = sum(0.4 - abs(d - 0.75) for _, d in rows)
            return _s(g)

        r = combine.greedy_search(score, [12], ["q"])
        self.assertEqual(dict(r["rows"])[12], 0.75)

    def test_call_cap_stops_search(self):
        def score(rows, prompts):
            return _s(0.01 * len(rows))

        r = combine.greedy_search(score, list(range(8)), ["q"], cap=3)
        self.assertTrue(r["capped"])
        self.assertLessEqual(r["calls"], 3)


class ProbeTest(unittest.TestCase):
    def test_broken_strength_ranks_under_clean_weakness(self):
        def score(rows, prompts):
            L = rows[0][0]
            if L == 9:
                return _s(0.9, mean=0.9, p10=0.05)  # strong, broken
            return _s(0.05)                          # weak, clean

        ranked, table = combine.probe_layers(score, [9, 12], ["q"])
        self.assertEqual(ranked, [12, 9])
        self.assertAlmostEqual(table[9]["probe_value"], -0.1)

    def test_clean_layers_rank_by_gain(self):
        gains = {12: 0.2, 10: 0.15}
        ranked, _ = combine.probe_layers(
            lambda rows, p: _s(gains[rows[0][0]]), [10, 12], ["q"])
        self.assertEqual(ranked, [12, 10])


class ConfirmTest(unittest.TestCase):
    def test_fresh_drop_fails(self):
        # The layer 12 case: strong on search prompts, weak on fresh.
        def score(rows, prompts):
            return _s(0.338 if prompts == ["search"] else 0.129)

        v = combine.confirm(score, ((12, 0.5),), ["fresh"])
        self.assertFalse(v["passed"])
        self.assertEqual(v["score"]["gain"], 0.129)

    def test_clean_strong_fresh_passes(self):
        v = combine.confirm(lambda rows, p: _s(0.45), ((12, 0.5),), ["f"])
        self.assertTrue(v["passed"])

    def test_confirm_records_the_spent_set(self):
        # A re-roll on the same set must be visible: the verdict
        # names the set by hash and size.
        v = combine.confirm(lambda rows, p: _s(0.45), ((12, 0.5),),
                            ["q one", "q two"])
        self.assertEqual(v["fresh_prompts"], 2)
        self.assertEqual(len(v["prompts_sha256"]), 16)
        other = combine.confirm(lambda rows, p: _s(0.45), ((12, 0.5),),
                                ["q one", "q three"])
        self.assertNotEqual(v["prompts_sha256"],
                            other["prompts_sha256"])

    def test_empty_mix_fails_without_scoring(self):
        def boom(rows, prompts):
            raise AssertionError("must not score an empty mix")

        self.assertFalse(combine.confirm(boom, (), ["f"])["passed"])


if __name__ == "__main__":
    unittest.main()
