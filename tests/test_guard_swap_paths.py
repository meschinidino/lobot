"""Guard swap proof: every guard path runs with LOBOT_GUARD=noop.

Each test calls the real entry point for one path. The noop guard
must not block. The verdict must carry the old report shape.
No weights. The engine objects are plain fakes.
"""
import unittest
from unittest import mock

import torch

from model_lobotomy import combine, followups, guards, server, trait_eval


class NoopSwapTest(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(
            "os.environ", {"LOBOT_GUARD": "noop"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_port_has_no_old_surface(self):
        for name in ("check", "check_verified", "check_confirmed",
                     "check_full", "check_screen"):
            self.assertFalse(hasattr(guards.ScreensV2Guard, name), name)
            self.assertFalse(hasattr(guards.NoopGuard, name), name)

    def test_run_chat_removal_verify_guard_entry(self):
        # Run, chat, removal and verify all call server.helpers._verified_guard.
        with mock.patch.object(server.helpers.guard_mod, "refusal_judge_fn",
                               return_value=lambda text: 1.0):
            gv = server.helpers._verified_guard(object(), object(), {}, True)
        self.assertFalse(gv.blocked)
        self.assertTrue(gv.verified)
        event = server.helpers._guard_event_dict(gv)
        self.assertFalse(event["blocked"])
        self.assertTrue(event["enforced"])
        self.assertIn("min_ratio", event)

    def test_create_guard_stage(self):
        ctx = {"eng": object(),
               "frozen": {"layer": 1, "unit": torch.tensor([1.0, 0.0]),
                          "dose": 0.5},
               "med": {1: 2.0}, "out": {}, "progress": None,
               "timings": {}, "seed": 0, "guard_flag": True,
               "research_mode": False, "cards": {}, "base_cache": mock.Mock(hits=0, misses=0)}
        with mock.patch.object(trait_eval.guard_mod, "refusal_judge_fn",
                               return_value=lambda text: 1.0):
            trait_eval._stage_guard(ctx)
        self.assertFalse(ctx["out"]["guard"]["blocked"])
        self.assertFalse(ctx["gv"].blocked)
        self.assertIn("min_ratio", ctx["out"]["guard"])

    def test_combine_check_set(self):
        patches = [{"direction": [1.0, 0.0],
                    "meta": {"layer": 1, "dose": 1.0, "scale": 1.0}}]
        with mock.patch.object(combine.guard_mod, "refusal_judge_fn",
                               return_value=lambda text: 1.0):
            out = combine.check_set(["a"], patches, object(), medians={},
                                    max_new_tokens=4, seed=0)
        self.assertFalse(out["blocked"])
        self.assertTrue(out["verified"])
        self.assertIn("min_ratio", out["guard"])

    def test_followup_guard_and_confirm(self):
        import tempfile
        from pathlib import Path
        seen = {}

        def fake_repair(direction, dose, **kw):
            seen["guard"] = kw["guard_fn"](direction, dose, ["refusal"])
            seen["confirm"] = kw["confirm_fn"](direction, dose)
            return {"status": "clean", "candidates": [], "flagged": [],
                    "chosen": None, "methods": [], "dose": dose}

        job = {"report": {"trait": "pirate", "seed": 0, "schedule": "fast",
                          "fingerprint": {"model_id": "fake"},
                          "splits": {"tuning": ["q"], "final": ["q"],
                                     "extraction": ["q"]}},
               "meta": {"layer": 1, "scale": 1.0, "dose": 0.5,
                        "report_card": {}},
               "direction": torch.tensor([1.0, 0.0])}
        eng = mock.Mock()
        eng.spec = object()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(followups.trait_eval, "load_job",
                                  return_value=job), \
                mock.patch.object(followups.trait_eval, "family_cfg",
                                  return_value={"judge": "x"}), \
                mock.patch.object(followups.trait_eval, "schedule_params",
                                  return_value={"tune_n": 1, "final_n0": 1,
                                                "final_nmax": 1,
                                                "gen_tokens": 4}), \
                mock.patch.object(followups.trait_eval, "TorchLane",
                                  return_value=mock.Mock(info=lambda: {})), \
                mock.patch.object(followups.guard_mod, "refusal_judge_fn",
                                  return_value=lambda text: 1.0), \
                mock.patch.object(followups.repair_mod, "repair",
                                  side_effect=fake_repair):
            followups.repair_job(eng, Path(tmp), lane=mock.Mock(info=lambda: {}))
        self.assertFalse(seen["guard"]["refusal"]["blocked"])
        self.assertFalse(seen["confirm"]["verdict"]["blocked"])
        self.assertTrue(seen["confirm"]["verdict"]["confirmed"])

    def test_stream_guard_two_steps(self):
        from model_lobotomy import server_helpers as helpers
        with mock.patch.object(helpers.guard_mod, "refusal_judge_fn",
                               return_value=lambda text: 1.0):
            events = list(helpers._verified_guard_stream(
                object(), object(), {}, True))
        stages = [e.decode() if isinstance(e, bytes) else str(e)
                  for e in events]
        self.assertTrue(any("guard_full" in s for s in stages), stages)

    def test_delete_check_guard_entry(self):
        # trait_eval.delete_check reads the port verdict report.
        gv = guards.get_guard().confirm(
            object(), object(), medians={}, max_new_tokens=4, seed=0,
            judge_fn=lambda text: 1.0, research_mode=True)
        self.assertFalse(gv.report["blocked"])


if __name__ == "__main__":
    unittest.main()
