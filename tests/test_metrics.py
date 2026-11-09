"""Metrics block: comparable, recomputable, hard to game. No weights."""
import json
import statistics
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from model_lobotomy import engine as engine_mod
from model_lobotomy import metrics as M
from model_lobotomy import prompt_bank
from model_lobotomy import trait_eval as T


def _report(**over):
    rep = {
        "trait": "pirate", "seed": 0, "schedule": "full",
        "fingerprint": {"model_id": "m", "revision": "r1"},
        "ports": {"guard": "screens", "judge": "local",
                  "direction": "diffmean", "dose_rule": "fluency-first-v3"},
        "lane": {"lane": "torch"},
        "judge": {"kind": "local", "model": "m", "agreement": None},
        "provenance": {"bank_version": "v1", "bank_sha256": "abc",
                       "gen_tokens": 60, "schedule": "full",
                       "schedule_params": {"gen_tokens": 60},
                       "judge_label_sha256": "def"},
        "frozen": {"layer": 3, "dose": 1.0, "tuning_gain": 0.8},
        "guard": {"enforced": True, "research_mode": False,
                  "blocked": False},
        "capture_cache_hit": False,
        "splits": {"final": [f"q{i}" for i in range(30)]},
        "early_stop": {"update_up": {"n": 30, "stopped_early": False}},
        "final": {"update_up": {"gains": [0.5] * 30},
                  "update_down": {"gains": [0.4] * 30},
                  "delete": {"gains": [0.3] * 30},
                  "control_gains": [0.0] * 30,
                  "control_removal_gains": [0.0] * 30,
                  "control_delete_gains": [0.0] * 30},
        "metrics": {"work": {"gen_batches": 10, "gen_prompts": 300,
                             "new_tokens": 9000, "score_pairs": 600,
                             "nll_pairs": 200}},
        "seconds": 100.0,
        "timings": {"extract": 10.0, "tune": 40.0, "final": 30.0},
    }
    rep.update(over)
    return rep


class BankHashTest(unittest.TestCase):
    def test_bank_sha_matches_module_constant(self):
        self.assertEqual(M.bank_sha256(prompt_bank.PROMPT_BANK),
                         prompt_bank.BANK_SHA256)

    def test_bank_sha_changes_when_a_prompt_changes(self):
        edited = list(prompt_bank.PROMPT_BANK)
        edited[0] = "an easier prompt"
        self.assertNotEqual(M.bank_sha256(edited), prompt_bank.BANK_SHA256)


class PairedEffectTest(unittest.TestCase):
    def test_paired_effect_recomputes_from_raw_gains(self):
        trait = [0.6, 0.4, 0.5, 0.5]
        control = [0.1, 0.0, 0.2, 0.1]
        out = M.paired_effect(trait, control)
        self.assertEqual(out["n"], 4)
        self.assertAlmostEqual(
            out["mean"],
            statistics.fmean([t - c for t, c in zip(trait, control)]))
        self.assertEqual(out["win_rate"], 1.0)
        self.assertLessEqual(out["ci"][0], out["mean"])
        self.assertLessEqual(out["mean"], out["ci"][1])

    def test_paired_effect_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            M.paired_effect([0.5, 0.5], [0.1])


class SideEffectsTest(unittest.TestCase):
    def test_side_effects_count_refusal_empty_and_repetition(self):
        pairs = [
            {"q": "q1", "base": "I cannot help with that",
             "steered": "a clear helpful answer with several words"},
            {"q": "q2", "base": "a clear helpful answer with several words",
             "steered": ""},
            {"q": "q3", "base": "one two three four five six seven eight",
             "steered": "loop loop loop loop loop loop loop loop"},
            {"q": "q4", "base": "short answer",
             "steered": "a much longer steered answer with many more words"},
        ]
        out = M.text_side_effects(pairs, refusal_fn=T.refusal_gate)
        self.assertEqual(out["n"], 4)
        self.assertEqual(out["refusal"]["base_rate"], 0.25)
        self.assertEqual(out["refusal"]["steered_rate"], 0.0)
        self.assertEqual(out["refusal"]["delta"], -0.25)
        self.assertEqual(out["empty"]["steered_rate"], 0.25)
        self.assertEqual(out["repetition"]["steered_rate"], 0.25)
        self.assertGreater(out["length_words"]["steered_mean"], 0)

    def test_side_effects_without_refusal_fn_says_missing_not_zero(self):
        out = M.text_side_effects(
            [{"q": "q", "base": "x y", "steered": "x y z"}])
        self.assertIsNone(out["refusal"])


class ComparabilityTest(unittest.TestCase):
    def test_clean_full_run_is_headline_eligible(self):
        out = M.comparability(_report())
        self.assertTrue(out["quality_headline_eligible"],
                        out["quality_reasons"])
        self.assertTrue(out["speed_headline_eligible"],
                        out["speed_reasons"])
        self.assertEqual(out["reasons"], [])

    def test_missing_evidence_fails_closed(self):
        out = M.comparability({})
        self.assertFalse(out["quality_headline_eligible"])
        self.assertIn("missing_guard_result", out["quality_reasons"])
        self.assertIn("missing_frozen_candidate",
                      out["quality_reasons"])
        self.assertIn("missing_final_gains:update_up",
                      out["quality_reasons"])

    def test_one_gain_labeled_n30_does_not_qualify(self):
        rep = _report()
        rep["final"] = dict(rep["final"])
        rep["final"]["update_up"] = {"gains": [0.9]}
        rep["early_stop"] = {"update_up": {"n": 30,
                                           "stopped_early": False}}
        out = M.comparability(rep)
        self.assertFalse(out["quality_headline_eligible"])
        self.assertIn("final_n_below_30", out["quality_reasons"])

    def test_stale_stored_metrics_are_recomputed_in_compare(self):
        import importlib.util
        root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "compare_runs3", root / "scripts" / "compare_runs.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        a = _report()
        b = _report()
        b["metrics"] = dict(b["metrics"])
        b["metrics"]["verbs"] = {
            "update_up": {"paired": {"mean": 99.0, "ci": [99.0, 99.0],
                                     "n": 30}}}
        code, lines = mod.compare(a, b)
        self.assertEqual(code, 0, lines)
        self.assertFalse(any("99" in ln for ln in lines), lines)

    def test_control_deterioration_is_named(self):
        import importlib.util
        root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "compare_runs4", root / "scripts" / "compare_runs.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        b = _report()
        b["final"] = dict(b["final"],
                          control_gains=[-0.4] * 30)
        code, lines = mod.compare(_report(), b)
        self.assertTrue(any("control got worse" in ln for ln in lines),
                        lines)

    def test_fast_schedule_is_provisional(self):
        out = M.comparability(_report(schedule="fast"))
        self.assertFalse(out["quality_headline_eligible"])
        self.assertIn("schedule_not_full", out["reasons"])

    def test_fake_judge_is_simulated(self):
        rep = _report()
        rep["ports"] = dict(rep["ports"], judge="fake")
        out = M.comparability(rep)
        self.assertFalse(out["quality_headline_eligible"])
        self.assertIn("simulated_port:judge=fake", out["reasons"])

    def test_unenforced_guard_blocks_quality_headline(self):
        rep = _report()
        rep["guard"] = {"enforced": False, "research_mode": False}
        out = M.comparability(rep)
        self.assertFalse(out["quality_headline_eligible"])
        self.assertIn("guard_not_enforced", out["reasons"])

    def test_early_stop_and_small_n_block_quality_headline(self):
        rep = _report()
        rep["final"] = dict(rep["final"])
        rep["final"]["update_up"] = {"gains": [0.5] * 8}
        rep["early_stop"] = {"update_up": {"n": 8, "stopped_early": True}}
        out = M.comparability(rep)
        self.assertFalse(out["quality_headline_eligible"])
        self.assertIn("stopped_early", out["reasons"])
        self.assertIn("final_n_below_30", out["reasons"])

    def test_warm_capture_cache_blocks_speed_but_not_quality(self):
        out = M.comparability(_report(capture_cache_hit=True))
        self.assertTrue(out["quality_headline_eligible"], out["reasons"])
        self.assertFalse(out["speed_headline_eligible"])
        self.assertIn("capture_cache_hit", out["reasons"])

    def test_compare_reports_names_differing_keys(self):
        a, b = _report(), _report()
        self.assertTrue(M.compare_reports(a, b)["comparable"])
        b["provenance"] = dict(b["provenance"], bank_sha256="other")
        out = M.compare_reports(a, b)
        self.assertFalse(out["comparable"])
        self.assertIn("bank_sha256", out["differing"])


class BuildMetricsTest(unittest.TestCase):
    def test_build_metrics_recomputes_and_accounts_time(self):
        rep = _report()
        rep["frozen"] = {"layer": 3, "dose": 1.0, "tuning_gain": 0.8}
        block = M.build_metrics(rep, elapsed=100.0)
        self.assertEqual(block["version"], M.METRICS_VERSION)
        up = block["verbs"]["update_up"]
        self.assertAlmostEqual(up["paired"]["mean"], 0.5)
        self.assertAlmostEqual(block["winners_curse_gap"], 0.3)
        self.assertAlmostEqual(
            block["time"]["unaccounted_seconds"], 20.0)
        self.assertAlmostEqual(block["time"]["accounted_share"], 0.8)


class LengthAssertTest(unittest.TestCase):
    def test_loop_gains_fails_on_short_lane_output(self):
        class ShortLane:
            def gen_many(self, prompts, edits, max_new, cancel=None):
                return ["only one"]

            def score_many(self, pairs, label):
                return [0.5] * len(pairs)

        with self.assertRaises(ValueError):
            T._loop_gains(None, None, {"judge": "x"}, ["q1", "q2"],
                          engine_mod.EMPTY_EDITS, 1.0,
                          {"gen_tokens": 4}, None, None, None, "final",
                          lane=ShortLane())


class VerdictStageTest(unittest.TestCase):
    def _ctx(self, tmp, up_passed, down_passed):
        def card(passed, mean):
            return {"passed": passed,
                    "rules": {"trait_ci_excludes_zero": passed},
                    "trait_ci": (mean - 0.1, mean + 0.1),
                    "trait_mean": mean,
                    "control_ci": (-0.05, 0.05),
                    "paired_ci": (mean - 0.1, mean + 0.1)}

        workdir = Path(tmp) / "w"
        workdir.mkdir(parents=True)
        out = {"trait": "pirate", "read_gate": {"pass": False},
               "fingerprint": {"model_id": "fake"}, "guard": {},
               "timings": {}}
        return {
            "trait_name": "pirate", "seed": 0, "workdir": workdir,
            "out": out,
            "frozen": {"layer": 1, "dose": 1.0,
                       "unit": torch.ones(8) / 8 ** 0.5},
            "med": {1: 1.0}, "fin": ["q1"], "t0": time.perf_counter(),
            "cards": {"update_up": card(up_passed, 0.5),
                      "update_down": card(down_passed, 0.7),
                      "delete": card(False, 0.0)},
            "gv": SimpleNamespace(blocked=False),
            "cfg": {"pos": "be pirate", "judge": "pirate-like"},
            "params": {"gen_tokens": 60}, "schedule": "full",
        }

    def test_two_eligible_verbs_do_not_crash_the_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._ctx(tmp, up_passed=True, down_passed=True)
            T._stage_verdict(ctx)
            self.assertEqual(ctx["out"]["verdict"], "knob")
            self.assertEqual(
                sorted(ctx["out"]["ledger"]["recorded"]),
                ["delete", "update_down", "update_up"])

    def test_patch_carries_the_earning_verbs_card(self):
        from model_lobotomy import patch as patch_mod
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._ctx(tmp, up_passed=False, down_passed=True)
            T._stage_verdict(ctx)
            loaded = patch_mod.load_patch(
                ctx["workdir"] / ctx["out"]["patch"])
            self.assertAlmostEqual(
                loaded["meta"]["report_card"]["trait_mean"], 0.7)
            self.assertEqual(loaded["meta"]["recipe"]["verb"],
                             "update_down")
            self.assertEqual(loaded["meta"]["recipe"]["gen_tokens"], 60)


class FakeEng:
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
        tag = "base" if edits == engine_mod.EMPTY_EDITS else "pirate steered"
        return {"text": f"{tag} reply to {prompt}"}


class ReportWiringTest(unittest.TestCase):
    def test_run_trait_writes_metrics_and_provenance(self):
        eng = FakeEng()
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
        prov = rep["provenance"]
        self.assertEqual(prov["gen_tokens"], 24)
        self.assertEqual(prov["n_final"], 30)
        self.assertEqual(prov["n_reserve"], 10)
        self.assertTrue(prov["bank_sha256"])
        self.assertEqual(rep["judge"]["kind"], "fake")
        if rep.get("frozen"):
            block = rep["metrics"]
            self.assertEqual(block["version"], M.METRICS_VERSION)
            comp = block["comparability"]
            self.assertFalse(comp["quality_headline_eligible"])
            self.assertIn("simulated_port:judge=fake", comp["reasons"])
            self.assertIn("schedule_not_full", comp["reasons"])


class CompareScriptTest(unittest.TestCase):
    def test_compare_refuses_ineligible_runs(self):
        import importlib.util
        root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "compare_runs", root / "scripts" / "compare_runs.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        code, lines = mod.compare(_report(schedule="fast"), _report())
        self.assertEqual(code, 2)
        self.assertTrue(any("not comparable" in ln for ln in lines))

    def test_compare_prints_deltas_for_clean_runs(self):
        import importlib.util
        root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "compare_runs2", root / "scripts" / "compare_runs.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        code, lines = mod.compare(_report(), _report(seconds=80.0))
        self.assertEqual(code, 0, lines)
        self.assertTrue(any("update_up: trait" in ln for ln in lines))


class LaneBatchTest(unittest.TestCase):
    def _lane(self, eng):
        lane = T.TorchLane.__new__(T.TorchLane)
        lane.eng = eng
        lane._judged = {}
        lane._nlls = {}
        lane._tokens_complete = True
        lane._work = {"gen_batches": 0, "gen_prompts": 0,
                      "new_tokens": None, "score_pairs": 0,
                      "nll_pairs": 0}
        return lane

    def test_gen_many_uses_run_many_when_available(self):
        class BatchEng:
            def run_many(self, prompts, edits, medians=None,
                         max_new_tokens=None, seed=0, cancel=None):
                return [{"text": f"r:{q}", "new_tokens": 3}
                        for q in prompts]

        lane = self._lane(BatchEng())
        out = lane.gen_many(["a", "b", "c"],
                            engine_mod.EMPTY_EDITS, 8)
        self.assertEqual(out, ["r:a", "r:b", "r:c"])
        self.assertEqual(lane.counters()["gen_batches"], 1)
        self.assertEqual(lane.counters()["new_tokens"], 9)

    def test_gen_many_fails_on_short_batch(self):
        class ShortEng:
            def run_many(self, prompts, edits, medians=None,
                         max_new_tokens=None, seed=0, cancel=None):
                return [{"text": "only", "new_tokens": 1}]

        lane = self._lane(ShortEng())
        with self.assertRaises(ValueError):
            lane.gen_many(["a", "b"], engine_mod.EMPTY_EDITS, 8)


class LaneCounterTest(unittest.TestCase):
    def test_torch_lane_counts_prompts_and_tokens(self):
        class Eng:
            def run(self, q, edits, medians=None, max_new_tokens=None,
                    seed=0, cancel=None):
                return {"text": "ok", "new_tokens": 7}

        lane = T.TorchLane.__new__(T.TorchLane)  # counter init below
        lane.eng = Eng()
        lane._judged = {}
        lane._nlls = {}
        lane._tokens_complete = True
        lane._work = {"gen_batches": 0, "gen_prompts": 0,
                      "new_tokens": None, "score_pairs": 0,
                      "nll_pairs": 0}
        out = lane.gen_many(["a", "b"], engine_mod.EMPTY_EDITS, 8)
        self.assertEqual(out, ["ok", "ok"])
        self.assertEqual(lane.counters()["gen_prompts"], 2)
        self.assertEqual(lane.counters()["new_tokens"], 14)


if __name__ == "__main__":
    unittest.main()
