"""ML v3 in the app: learned-mix knob recognition and edit math."""
import json
import tempfile
import unittest
from pathlib import Path

from model_lobotomy import server_helpers as helpers


class LearnedPassTest(unittest.TestCase):
    def _dir(self, payload):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = Path(tmp.name) / "learned.json"
        if payload is not None:
            p.write_text(json.dumps(payload))
        return Path(tmp.name)

    def test_v3b_pass_counts(self):
        d = self._dir({"family": "be-rude-hateful-and-mocking",
                       "best_scale": 0.65,
                       "confirm_v3b": {"passed": True, "scale": 0.625,
                                       "set": "fresh_questions_v4.json",
                                       "facets": ["rude", "hateful", "mocking"],
                                       "score": {
                           "gain": 0.402, "fluency_mean": 0.837,
                           "fluency_p10": 0.392}}})
        out = helpers._learned_pass(d)
        # The served scale is the confirmed scale, not best_scale.
        self.assertEqual(out["scale"], 0.625)
        self.assertEqual(out["gain"], 0.402)

    def test_confirm_under_gain_bar_does_not_count(self):
        d = self._dir({"family": "be-rude-hateful-and-mocking",
                       "confirm_v3b": {"passed": True, "scale": 0.625,
                                       "set": "fresh.json",
                                       "facets": ["rude", "hateful", "mocking"],
                                       "score": {"gain": 0.29,
                                                 "fluency_mean": 0.9,
                                                 "fluency_p10": 0.9}}})
        self.assertIsNone(helpers._learned_pass(d))

    def test_confirm_under_fluency_tail_bar_does_not_count(self):
        d = self._dir({"family": "be-rude-hateful-and-mocking",
                       "confirm_v3b": {"passed": True, "scale": 0.625,
                                       "set": "fresh.json",
                                       "facets": ["rude", "hateful", "mocking"],
                                       "score": {"gain": 0.4,
                                                 "fluency_mean": 0.9,
                                                 "fluency_p10": 0.29}}})
        self.assertIsNone(helpers._learned_pass(d))

    def test_partial_facets_do_not_count(self):
        d = self._dir({"family": "be-rude-hateful-and-mocking",
                       "confirm_v3b": {"passed": True, "scale": 0.625,
                                       "set": "fresh.json",
                                       "facets": ["rude"],
                                       "score": {"gain": 0.4,
                                                 "fluency_mean": 0.9,
                                                 "fluency_p10": 0.9}}})
        self.assertIsNone(helpers._learned_pass(d))

    def test_confirm_without_scale_does_not_count(self):
        d = self._dir({"family": "be-rude-hateful-and-mocking",
                       "confirm_v3b": {"passed": True,
                                       "set": "fresh.json",
                                       "facets": ["rude", "hateful", "mocking"],
                                       "score": {"gain": 0.4,
                                                 "fluency_mean": 0.9,
                                                 "fluency_p10": 0.9}}})
        self.assertIsNone(helpers._learned_pass(d))

    def test_plain_confirm_without_set_does_not_count(self):
        # The v3 search confirm failed; a pass with no fresh-set
        # name is not trusted.
        d = self._dir({"best_scale": 0.75,
                       "confirm": {"passed": True, "score": {}}})
        self.assertIsNone(helpers._learned_pass(d))

    def test_guard_blocked_flag_surfaces(self):
        d = self._dir({"family": "be-rude-hateful-and-mocking",
                       "confirm_v3b": {"passed": True, "scale": 0.625,
                                       "set": "fresh.json",
                                       "facets": ["rude", "hateful", "mocking"],
                                       "score": {"gain": 0.4,
                                                 "fluency_mean": 0.9,
                                                 "fluency_p10": 0.9}},
                       "guard": {"blocked": True}})
        self.assertTrue(helpers._learned_pass(d)["guard_blocked"])

    def test_old_judge_pass_does_not_count(self):
        # A pass with no facets ran under the single-question judge.
        d = self._dir({"best_scale": 0.625,
                       "confirm_v3b": {"passed": True, "score": {
                           "gain": 0.402, "fluency_p10": 0.392}}})
        self.assertIsNone(helpers._learned_pass(d))

    def test_failed_confirm_does_not_count(self):
        d = self._dir({"confirm_v3b": {"passed": False, "score": {}}})
        self.assertIsNone(helpers._learned_pass(d))

    def test_missing_file(self):
        self.assertIsNone(helpers._learned_pass(self._dir(None)))


class TakedownTest(unittest.TestCase):
    def test_taken_down_record_reads(self):
        from model_lobotomy import jobs_store
        with tempfile.TemporaryDirectory() as tmp:
            paths = jobs_store.JobPaths("j1", Path(tmp))
            self.assertIsNone(helpers._taken_down(paths))
            paths.taken_down.parent.mkdir(parents=True)
            paths.taken_down.write_text(json.dumps(
                {"reason": "failed its 50-question confirm"}))
            self.assertEqual(helpers._taken_down(paths)["reason"],
                             "failed its 50-question confirm")

    def test_taken_down_newest_job_earns_no_knob(self):
        from unittest import mock
        from model_lobotomy import jobs_store
        with tempfile.TemporaryDirectory() as tmp:
            jobs_dir = Path(tmp)
            job = {"id": "pirate-9", "trait": "pirate",
                   "status": "done", "updated": 10,
                   "model_id": helpers.engine._active_model_id,
                   "frozen": {"layer": 16, "dose": 0.5}}
            paths = jobs_store.JobPaths("pirate-9", jobs_dir)
            paths.trait.workdir.mkdir(parents=True)
            paths.report.write_text(json.dumps({
                "read_gate": {"pass": True},
                "final": {"update_up": {"card": {"passed": True}}}}))
            import torch
            from safetensors.torch import save_file
            save_file({"direction": torch.ones(8)},
                      str(paths.direction))
            paths.taken_down.write_text(json.dumps(
                {"reason": "gain 0.225 at 50 fresh questions"}))

            class Runner:
                def list_jobs(self):
                    return [job]

            runner = Runner()
            runner.jobs_dir = jobs_dir
            with mock.patch.object(helpers.STATE, "get_runner",
                                   return_value=runner):
                self.assertIsNone(helpers._latest_knob("pirate"))


class ResearchGateTest(unittest.TestCase):
    def test_page_cannot_turn_guard_off_without_owner_switch(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(helpers._effective_research(True))
        with mock.patch.dict(
                os.environ, {"LOBOT_ALLOW_RESEARCH": "1"}, clear=True):
            self.assertTrue(helpers._effective_research(True))
            self.assertFalse(helpers._effective_research(False))

    def test_research_run_writes_audit_record(self):
        import tempfile
        from unittest import mock
        from model_lobotomy import audit

        class GV:
            would_block = True
            blocked = False
            research_mode = True

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "audit.jsonl"
            with mock.patch.dict(
                    "os.environ", {"LOBOT_AUDIT_LOG": str(log)}):
                helpers._audit_research("run", "some-trait", GV())
            rows = audit.read(log)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event"], "research_run")
        self.assertTrue(rows[0]["would_block"])


class DialErrorTest(unittest.TestCase):
    def test_learned_trait_says_no_dial_not_missing(self):
        from unittest import mock
        with mock.patch.object(
                helpers, "_latest_knob",
                return_value={"kind": "learned", "vectors": {3: {}}}):
            msg = helpers._dial_error("some-trait")
        self.assertIn("No read dial for learned mixes", msg)
        self.assertNotIn("Create the trait first", msg)

    def test_unknown_trait_keeps_create_message(self):
        from unittest import mock
        with mock.patch.object(helpers, "_latest_knob",
                               return_value=None):
            self.assertIn("Create the trait first",
                          helpers._dial_error("nope"))


class KnobEarnedTest(unittest.TestCase):
    def test_unreadable_report_earns_no_knob(self):
        from model_lobotomy import jobs_store
        with tempfile.TemporaryDirectory() as tmp:
            paths = jobs_store.JobPaths("missing-job", Path(tmp))
            self.assertFalse(helpers._knob_earned(paths))


class LearnedVectorEditTest(unittest.TestCase):
    def test_edits_rebuild_the_tested_mix(self):
        knob = {"kind": "learned", "shipped_scale": 0.625,
                "vectors": {3: {"unit": [1.0, 0.0], "norm": 2.0},
                            7: {"unit": [0.0, 1.0], "norm": 4.0}}}
        edits = helpers._trait_vector(knob, 1.0, 0.0)
        self.assertEqual(len(edits), 2)
        by_layer = {e.layer: e for e in edits}
        # dose 1.0 = the tested mix: norm * shipped scale.
        self.assertAlmostEqual(by_layer[3].scale, 1.25)
        self.assertAlmostEqual(by_layer[7].scale, 2.5)
        edits2 = helpers._trait_vector(knob, 0.5, 0.0)
        self.assertAlmostEqual(
            {e.layer: e for e in edits2}[3].scale, 0.625)


if __name__ == "__main__":
    unittest.main()
