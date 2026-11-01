"""GR tests: projection, fit-set hygiene, candidate order and choice."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_lobotomy import guard as guard_mod  # noqa: E402
from model_lobotomy import guard_screens as S  # noqa: E402
from model_lobotomy import repair as R  # noqa: E402


class ProjectTest(unittest.TestCase):
    def test_result_is_orthogonal_unit(self):
        g = torch.Generator().manual_seed(0)
        v = torch.randn(64, generator=g)
        safety = [torch.randn(64, generator=g) for _ in range(2)]
        unit, info = R.project_out(v, safety)
        self.assertAlmostEqual(float(unit.norm()), 1.0, places=5)
        for s in safety:
            self.assertAlmostEqual(float(unit @ s), 0.0, places=4)
        self.assertGreater(info["removed_share"], 0)
        self.assertLess(info["cos_to_original"], 1.0)

    def test_rows_are_all_removed(self):
        g = torch.Generator().manual_seed(1)
        v = torch.randn(32, generator=g)
        rows = torch.randn(3, 32, generator=g)
        single = torch.randn(32, generator=g)
        unit, _ = R.project_out(v, [rows, single])
        for r in [*rows, single]:
            self.assertAlmostEqual(float(unit @ r), 0.0, places=4)

    def test_no_safety_is_identity(self):
        v = torch.tensor([3.0, 4.0])
        unit, info = R.project_out(v, [])
        self.assertTrue(torch.allclose(unit, torch.tensor([0.6, 0.8])))
        self.assertEqual(info["removed_share"], 0.0)

    def test_inside_span_raises(self):
        with self.assertRaises(ValueError):
            R.project_out(torch.tensor([1.0, 0.0]),
                          [torch.tensor([2.0, 0.0])])


class FitSetTest(unittest.TestCase):
    def test_disjoint_from_guard_prompts(self):
        guard = set(guard_mod.REFUSAL_SET) | set(S.HARMFUL_SET)
        self.assertFalse(guard & set(R.REFUSAL_FIT))
        self.assertFalse(set(S.HONESTY_SET) & set(R.HONESTY_FIT))
        secrets = {p["secret"] for p in S.PRIVACY_PROMPTS}
        self.assertFalse(secrets & {s for _, s in R.PRIVACY_FIT})

    def test_sides_are_paired(self):
        for screen in ("refusal", "honesty", "privacy"):
            safe, unsafe = R.fit_messages(screen)
            self.assertEqual(len(safe), len(unsafe))
            self.assertTrue(all(m[-1]["role"] == "user" for m in safe))
        safe, unsafe = R.fit_messages("privacy")
        self.assertIn("Never reveal", safe[0][0]["content"])
        self.assertIn("may share", unsafe[0][0]["content"])
        with self.assertRaises(ValueError):
            R.fit_messages("nope")

    def test_safety_direction_is_unit_mean_diff(self):
        def cap(msgs_list, layer):
            return torch.tensor([
                [2.0 if m[0]["content"] in R.REFUSAL_FIT else 0.0, 1.0]
                for m in msgs_list])

        d = R.safety_direction(cap, "refusal", 5)
        self.assertTrue(torch.allclose(d, torch.tensor([1.0, 0.0])))


def verdict(*flagged_screens, screens=("refusal", "privacy")):
    return {s: {"blocked": False, "would_block": s in flagged_screens}
            for s in screens}


class RepairTest(unittest.TestCase):
    def setUp(self):
        self.v = torch.tensor([1.0, 1.0, 0.0])
        self.safety = torch.tensor([1.0, 0.0, 0.0])
        self.calls = []

    def run_repair(self, clean_at, gain=0.3, card_pass=True,
                   after_flags=()):
        """clean_at: set of (method, factor) the guard passes."""
        v = self.v / self.v.norm()

        def method_of(unit):
            return "raw" if torch.allclose(unit, v) else "project"

        def guard_fn(unit, dose, screens):
            key = (method_of(unit), round(dose / 2.0, 2))
            self.calls.append(("guard", key, tuple(screens)))
            ok = key in clean_at
            return {s: {"would_block": not ok} for s in screens}

        def confirm_fn(unit, dose):
            self.calls.append(("confirm", method_of(unit), dose))
            if dose == 2.0 and method_of(unit) == "raw":
                return {"screens": verdict("refusal"), "verdict": {}}
            return {"screens": verdict(*after_flags), "verdict": {}}

        return R.repair(
            self.v, 2.0,
            safety_fn=lambda s: self.safety,
            guard_fn=guard_fn,
            tuning_gain_fn=lambda u, d: [gain, gain],
            confirm_fn=confirm_fn,
            final_card_fn=lambda u, d: {"passed": card_pass,
                                        "trait_mean": 0.2})

    def test_first_clean_candidate_wins(self):
        rec = self.run_repair({("raw", 0.75), ("project", 0.5)})
        self.assertEqual(rec["status"], "repaired")
        self.assertEqual((rec["chosen"]["method"], rec["chosen"]["factor"]),
                         ("raw", 0.75))
        tried = [c[1] for c in self.calls if c[0] == "guard"]
        self.assertEqual(tried, [("project", 1.0), ("project", 0.75),
                                 ("raw", 0.75)])
        # Only the flagged screen is screened per candidate.
        self.assertTrue(all(c[2] == ("refusal",) for c in self.calls
                            if c[0] == "guard"))
        self.assertEqual(rec["flagged"], ["refusal"])
        self.assertAlmostEqual(rec["safety_cos"]["refusal"], 2 ** -0.5,
                               places=5)

    def test_project_removes_safety_component(self):
        rec = self.run_repair({("project", 1.0)})
        self.assertEqual(rec["chosen"]["method"], "project")
        self.assertAlmostEqual(float(rec["unit"] @ self.safety), 0.0)
        self.assertAlmostEqual(rec["projection"]["removed_share"], 0.5,
                               places=5)

    def test_unrepaired_when_none_pass(self):
        rec = self.run_repair(set())
        self.assertEqual(rec["status"], "unrepaired")
        self.assertIsNone(rec["chosen"])
        self.assertEqual(len(rec["candidates"]), len(R.CANDIDATES))

    def test_zero_gain_is_not_a_repair(self):
        rec = self.run_repair({("project", 1.0)}, gain=0.0)
        self.assertEqual(rec["status"], "unrepaired")
        self.assertFalse(rec["candidates"][0]["ok"])

    def test_chosen_must_pass_card_and_full_guard(self):
        rec = self.run_repair({("project", 1.0)}, card_pass=False)
        self.assertEqual(rec["status"], "unrepaired")
        self.assertIsNotNone(rec["chosen"])
        rec = self.run_repair({("project", 1.0)}, after_flags=("privacy",))
        self.assertEqual(rec["status"], "unrepaired")

    def test_gradient_candidates_use_projected_rows(self):
        rows = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        seen = []
        rec = R.repair(
            self.v, 2.0, safety_fn=lambda s: rows,
            guard_fn=lambda u, d, sc: (seen.append((u, d)) or
                                       {s: {"would_block": False}
                                        for s in sc}),
            tuning_gain_fn=lambda u, d: [0.2],
            confirm_fn=lambda u, d: {
                "screens": verdict("refusal") if d == 2.0 and
                float(u[0]) > 0.1 else verdict(), "verdict": {}},
            final_card_fn=lambda u, d: {"passed": True},
            candidates=R.GRADIENT_CANDIDATES)
        self.assertEqual(rec["status"], "repaired")
        self.assertEqual(rec["chosen"]["method"], "gradient")
        self.assertTrue(torch.allclose(seen[0][0],
                                       torch.tensor([0.0, 1.0, 0.0])))
        self.assertEqual(rec["safety_rows"], {"refusal": 2})
        self.assertEqual(rec["methods"], ["gradient"])

    def test_safe_targets_cover_screens(self):
        self.assertTrue(set(S.ALL_SCREENS) <= set(R.SAFE_TARGETS))
        for screen in ("refusal", "honesty", "privacy"):
            self.assertTrue(R.gradient_messages(screen))
        self.assertNotIn(R.HONESTY_SAFE,
                         R.gradient_messages("honesty")[0][0]["content"])

    def test_clean_patch_is_left_alone(self):
        rec = R.repair(
            self.v, 1.0, safety_fn=None, guard_fn=None,
            tuning_gain_fn=None,
            confirm_fn=lambda u, d: {"screens": verdict(), "verdict": {}},
            final_card_fn=None)
        self.assertEqual(rec["status"], "clean")
        self.assertNotIn("unit", R.record_json(rec))


if __name__ == "__main__":
    unittest.main()
