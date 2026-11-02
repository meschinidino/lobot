"""Tune with Muse: fake completer and fake engine. No API, no weights."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_lobotomy import audit as audit_mod  # noqa: E402
from model_lobotomy import muse as muse_mod  # noqa: E402
from model_lobotomy import tune as tune_mod  # noqa: E402


class Scripted:
    """complete(prompt) for Muse tune prompts. Records prompts."""

    def __init__(self, decisions, summaries=None):
        self.prompts = []
        self.decisions = list(decisions)
        self.summaries = list(summaries or [
            "The final gain was 0.40. The guard did not block it."])
        self.model = "fake-muse"
        self.usage = {"calls": 0}

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if prompt.startswith("You tune"):
            return self.decisions.pop(0) if len(self.decisions) > 1 \
                else self.decisions[0]
        if prompt.startswith("Write the final summary"):
            return self.summaries.pop(0) if len(self.summaries) > 1 \
                else self.summaries[0]
        raise AssertionError(prompt[:40])


def _try(gain=0.4, guard_pass=True, seconds=0.1):
    def try_fn(dose):
        return {"gain": gain, "guard_pass": guard_pass,
                "guard": {"blocked": not guard_pass,
                          "would_block": not guard_pass},
                "pairs": [{"q": "q", "plain": "a", "edited": "b",
                           "gain": gain}],
                "seconds": seconds}
    return try_fn


def _final(gain=0.4, blocked=False, seconds=0.2):
    def final_fn(dose):
        return {"gain": gain,
                "guard": {"blocked": blocked, "would_block": blocked},
                "seconds": seconds}
    return final_fn


TRY = json.dumps({"action": "try_dose", "dose": 0.5,
                  "reason": "start at the shipped dose",
                  "expected_gain": 0.3, "guard_pass": True})
STOP = json.dumps({"action": "stop", "dose": 0.5,
                   "reason": "the gain is high enough"})


class AuditIsolatedTest(unittest.TestCase):
    """Every agent run in this file writes to a temp audit log."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.audit_dir = Path(tmp.name)
        self.audit_log = self.audit_dir / "audit.jsonl"
        env = mock.patch.dict(
            "os.environ", {"LOBOT_AUDIT_LOG": str(self.audit_log)})
        env.start()
        self.addCleanup(env.stop)


class TuneAgentTest(AuditIsolatedTest):
    def test_real_audit_log_is_never_used(self):
        # This test fails when log_path() leaves the temp dir.
        self.assertTrue(str(audit_mod.log_path()).startswith(
            str(self.audit_dir)))
        self._agent(Scripted([STOP])).run("raise the gain", "patch-1")
        events = [r["event"] for r in audit_mod.read(self.audit_log)]
        self.assertEqual(events, ["tune.final"])

    def _agent(self, fake, audit_log=None, **kw):
        return tune_mod.TuneAgent(
            muse_mod.Muse(fake), kw.pop("try_fn", _try()),
            kw.pop("final_fn", _final()), audit_log=audit_log, **kw)

    def test_action_limit_is_five_tries(self):
        agent = self._agent(Scripted([TRY]))
        out = agent.run("raise the gain", "patch-1")
        self.assertEqual(len(out["tries"]), 5)
        self.assertEqual(out["steps"][-3]["step"], "stop")
        self.assertEqual(out["steps"][-3]["by"], "harness")
        self.assertEqual(out["final"]["dose"], 0.5)

    def test_prediction_logged_before_the_try(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log = Path(tmp.name) / "audit.jsonl"
        seen = {}

        def try_fn(dose):
            rows = audit_mod.read(log)
            seen["events"] = [r["event"] for r in rows]
            seen["dose"] = rows[-1]["dose"]
            return {"gain": 0.4, "guard_pass": True, "guard": {},
                    "pairs": [], "seconds": 0.1}

        agent = self._agent(Scripted([TRY, STOP]), audit_log=log,
                            try_fn=try_fn)
        agent.run("raise the gain", "patch-1")
        self.assertEqual(seen["events"], ["tune.prediction"])
        self.assertEqual(seen["dose"], 0.5)
        events = [r["event"] for r in audit_mod.read(log)]
        self.assertEqual(events, ["tune.prediction", "tune.measured",
                                  "tune.final"])

    def test_guard_block_is_reported(self):
        fake = Scripted([TRY, STOP], summaries=[
            "The final gain was 0.40. The guard blocked it."])
        agent = self._agent(fake, final_fn=_final(blocked=True))
        out = agent.run("raise the gain", "patch-1")
        self.assertEqual(out["status"], "failed")
        self.assertTrue(out["final"]["guard_blocked"])
        self.assertFalse(out["final"]["passed"])

    def test_summary_number_check_retries_then_passes(self):
        fake = Scripted([STOP], summaries=[
            "The final gain was 0.99. It passed.",
            "The final gain was 0.40. The guard did not block it."])
        out = self._agent(fake).run("raise the gain", "patch-1")
        self.assertIn("0.40", out["summary"])
        self.assertNotIn("0.99", out["summary"])

    def test_summary_number_check_fails_closed(self):
        fake = Scripted([STOP], summaries=[
            "The final gain was 0.99. It passed."])
        with self.assertRaises(muse_mod.MuseError):
            self._agent(fake).run("raise the gain", "patch-1")

    def test_confirmed_needs_both_halves(self):
        self.assertTrue(tune_mod.prediction_confirmed(0.3, True, 0.4, True))
        self.assertFalse(tune_mod.prediction_confirmed(0.3, True, 0.41, True))
        self.assertFalse(tune_mod.prediction_confirmed(0.3, True, 0.4, False))
        self.assertFalse(tune_mod.prediction_confirmed(0.3, True, -0.4, True))
        self.assertTrue(
            tune_mod.prediction_confirmed(-0.3, False, -0.4, False))

    def test_events_name_the_checks(self):
        out = self._agent(Scripted([TRY, STOP])).run(
            "raise the gain", "patch-1")
        measured = [s for s in out["steps"] if s["step"] == "measured"][0]
        self.assertEqual(measured["check"], "first-token check")
        self.assertEqual(out["final"]["check"], "full guard check")


class FakeEng:
    """Fake engine: edited answers say pirate, plain answers do not."""

    model = object()
    tokenizer = object()
    spec = None
    model_id = "meta-llama/x"
    revision = "rev"

    def run(self, prompt, edits, *, medians, max_new_tokens, seed):
        edited = bool(edits.vector)
        return {"text": "pirate answer" if edited else "plain answer"}


class TuneRouteWiringTest(AuditIsolatedTest):
    def _wiring(self, judge, fake):
        from model_lobotomy.guards import NoopGuard
        from model_lobotomy.routers import tune as tune_route
        rep = {"trait": "pirate",
               "splits": {"tuning": [f"t{i}" for i in range(8)],
                          "final": [f"h{i}" for i in range(8)]}}
        loaded = {"direction": [1.0, 0.0],
                  "meta": {"layer": 16, "dose": 0.5, "scale": 2.0}}
        patches = [
            mock.patch.object(tune_route.trait_eval, "judge_score", judge),
            mock.patch("model_lobotomy.guards.get_guard",
                       return_value=NoopGuard())]
        gen = tune_route._tune_events(
            FakeEng(), muse_mod.Muse(fake),
            mock.Mock(goal="raise the gain", patch_id="patch-1"),
            rep, loaded, "pirate voice", {})
        return gen, patches

    def test_first_event_arrives_before_the_run_ends(self):
        import time

        def slow_judge(model, tok, spec, q, reply, label):
            time.sleep(0.3)
            return 0.9 if reply.startswith("pirate") else 0.1

        fake = Scripted([TRY, STOP], summaries=[
            "The final gain was 0.80. The guard did not block it."])
        gen, patches = self._wiring(slow_judge, fake)
        with patches[0], patches[1]:
            start = time.time()
            first = json.loads(next(gen))
            elapsed = time.time() - start
            rest = [json.loads(c) for c in gen]
        self.assertEqual(first["stage"], "prediction")
        self.assertLess(elapsed, 0.3)
        self.assertEqual(rest[-1]["stage"], "done")

    def test_tune_events_with_fake_engine(self):
        from model_lobotomy.guards import NoopGuard
        from model_lobotomy.routers import tune as tune_route

        rep = {"trait": "pirate",
               "splits": {"tuning": [f"t{i}" for i in range(8)],
                          "final": [f"h{i}" for i in range(8)]}}
        loaded = {"direction": [1.0, 0.0],
                  "meta": {"layer": 16, "dose": 0.5, "scale": 2.0}}

        def judge(model, tok, spec, q, reply, label):
            return 0.9 if reply.startswith("pirate") else 0.1

        fake = Scripted([TRY, STOP], summaries=[
            "The final gain was 0.80. The guard did not block it."])
        events = []
        with mock.patch.object(tune_route.trait_eval, "judge_score", judge), \
                mock.patch("model_lobotomy.guards.get_guard",
                           return_value=NoopGuard()):
            gen = tune_route._tune_events(
                FakeEng(), muse_mod.Muse(fake),
                mock.Mock(goal="raise the gain", patch_id="patch-1"),
                rep, loaded, "pirate voice", {})
            for chunk in gen:
                events.append(json.loads(chunk))
        stages = [e["stage"] for e in events]
        self.assertEqual(stages, ["prediction", "measured", "stop",
                                  "final", "summary", "done"])
        measured = events[1]
        self.assertEqual(measured["gain"], 0.8)
        self.assertTrue(measured["guard_pass"])
        self.assertEqual(events[-1]["result"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
