"""Find with Muse: scripted tool calls and a fake engine. No API."""
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
from model_lobotomy import find as find_mod  # noqa: E402
from model_lobotomy import muse as muse_mod  # noqa: E402


def call(name, **args):
    return {"type": "function_call", "name": name,
            "arguments": json.dumps(args), "call_id": f"c{name}"}


class FakeMuse:
    """Scripted tool turns. Summaries go through the real check."""

    def __init__(self, turns, summaries=None):
        self.turns = list(turns)
        self.usage = {"calls": 0}
        texts = list(summaries or [
            "The search used 2 tool calls. The guard did not block "
            "the recipe."])
        self._summaries = muse_mod.Muse(
            lambda prompt: texts.pop(0) if len(texts) > 1 else texts[0])

    def tool_turn(self, items, tools):
        self.usage["calls"] += 1
        return self.turns.pop(0) if len(self.turns) > 1 else self.turns[0]

    def find_summary(self, summary):
        return self._summaries.find_summary(summary)

    def info(self):
        return {"model": "fake-muse", "usage": self.usage}


def _tools(**over):
    tools = {
        "find_neurons": lambda a: {"word": a["word"], "results": [
            {"layer": 20, "neuron": 123, "score": 3.1, "rank": 1}]},
        "read_neuron": lambda a: {"pushes_toward": {"up": []}},
        "measure_lift": lambda a: {"lift": 4.0, "words": a["words"]},
        "try_recipe": lambda a: {
            "plain": "I stayed home.", "edited": "I saw a storm.",
            "fork": {"index": 1}, "guard_pass": True, "guard": {}},
    }
    tools.update(over)
    return tools


def _final(blocked=False):
    return lambda rows: {"guard": {"blocked": blocked,
                                   "would_block": blocked}}


def _lift(value=4.0):
    return lambda rows, words: {"lift": value}


class AuditIsolatedTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.audit_dir = Path(tmp.name)
        self.audit_log = self.audit_dir / "audit.jsonl"
        env = mock.patch.dict(
            "os.environ", {"LOBOT_AUDIT_LOG": str(self.audit_log)})
        env.start()
        self.addCleanup(env.stop)


class FindAgentTest(AuditIsolatedTest):
    def _agent(self, fake, **kw):
        return find_mod.FindAgent(
            fake, kw.pop("tool_fns", _tools()),
            kw.pop("final_fn", _final()), kw.pop("lift_fn", _lift()),
            **kw)

    def test_real_audit_log_is_never_used(self):
        self.assertTrue(str(audit_mod.log_path()).startswith(
            str(self.audit_dir)))

    def test_happy_path_and_confirmed_lift(self):
        fake = FakeMuse([
            [call("find_neurons", word="storm", k=5)],
            [call("measure_lift", layer=20, neuron=123, dose=1.0,
                  words=["storm"], prediction=3.0)],
            [call("stop", rows=[{"layer": 20, "neuron": 123,
                                 "dose": 1.0}], words=["storm"],
                  reason="it lifts storm")],
        ], summaries=["The search used 3 tool calls. The lift was "
                      "4.0. The guard did not block the recipe."])
        out = self._agent(fake).run("bring up storms", "Tell me.")
        self.assertEqual(out["status"], "passed")
        self.assertEqual(out["tool_calls"], 3)
        measured = [s for s in out["steps"] if s["step"] == "result"
                    and s["tool"] == "measure_lift"][0]
        self.assertTrue(measured["confirmed"])

    def test_call_limit_is_twelve(self):
        fake = FakeMuse([[call("find_neurons", word="storm")]],
                        summaries=["The search used 12 tool calls. "
                                   "The guard did not block it."])
        out = self._agent(fake).run("bring up storms", "Tell me.")
        self.assertEqual(out["tool_calls"], 12)
        self.assertEqual(out["status"], "failed")

    def test_prediction_logged_before_measuring_call(self):
        seen = {}

        def measure(args):
            rows = audit_mod.read(self.audit_log)
            seen["events"] = [r["event"] for r in rows]
            return {"lift": 4.0}

        fake = FakeMuse([
            [call("measure_lift", layer=20, neuron=123, dose=1.0,
                  words=["storm"], prediction=3.0)],
            [call("stop", rows=[], words=[], reason="done")],
        ])
        self._agent(fake, tool_fns=_tools(measure_lift=measure),
                    audit_log=self.audit_log).run("g", "p")
        self.assertEqual(seen["events"], ["find.prediction"])
        events = [r["event"] for r in audit_mod.read(self.audit_log)]
        self.assertEqual(events[:2], ["find.prediction", "find.result"])

    def test_recipe_over_four_neurons_is_rejected(self):
        rows = [{"layer": 1, "neuron": i, "dose": 1.0} for i in range(5)]
        fake = FakeMuse([
            [call("try_recipe", rows=rows, prediction=["storm"])],
            [call("stop", rows=rows[:1], words=["storm"], reason="ok")],
        ], summaries=["The search used 2 tool calls. The lift was "
                      "4.0. The guard did not block the recipe."])
        out = self._agent(fake).run("g", "p")
        bad = [s for s in out["steps"] if s["step"] == "result"][0]
        self.assertIn("at most 4", bad["result"]["error"])
        self.assertEqual(len(out["final"]["rows"]), 1)

    def test_dose_out_of_range_is_rejected(self):
        fake = FakeMuse([
            [call("measure_lift", layer=20, neuron=123, dose=99.0,
                  words=["storm"], prediction=3.0)],
            [call("stop", rows=[], words=[], reason="done")],
        ])
        out = self._agent(fake).run("g", "p")
        bad = [s for s in out["steps"] if s["step"] == "result"][0]
        self.assertIn("outside", bad["result"]["error"])

    def test_guard_block_is_reported(self):
        fake = FakeMuse([
            [call("stop", rows=[{"layer": 20, "neuron": 123,
                                 "dose": 1.0}], words=["storm"],
                  reason="done")],
        ], summaries=["The search used 1 tool calls. The guard "
                      "blocked the recipe."])
        out = self._agent(fake, final_fn=_final(blocked=True)).run("g", "p")
        self.assertEqual(out["status"], "failed")
        self.assertTrue(out["final"]["guard_blocked"])
        self.assertFalse(out["final"]["passed"])

    def test_summary_number_check(self):
        fake = FakeMuse([
            [call("stop", rows=[], words=[], reason="done")],
        ], summaries=["The lift was 9.99 and it passed.",
                      "The search used 1 tool calls. The guard did "
                      "not block the recipe."])
        out = self._agent(fake).run("g", "p")
        self.assertNotIn("9.99", out["summary"])

    def test_lift_confirmed_factor_two(self):
        self.assertTrue(find_mod.lift_confirmed(3.0, 4.0))
        self.assertTrue(find_mod.lift_confirmed(3.0, 1.5))
        self.assertFalse(find_mod.lift_confirmed(3.0, 1.4))
        self.assertFalse(find_mod.lift_confirmed(3.0, 6.1))


class FakeEng:
    model = object()
    tokenizer = object()
    spec = None
    model_id = "meta-llama/x"
    revision = "rev"

    def run(self, prompt, edits, *, medians, max_new_tokens, seed):
        edited = bool(edits.neuron)
        return {"text": "I saw a storm." if edited
                else "I stayed home."}


class FindRouteWiringTest(AuditIsolatedTest):
    def test_find_events_with_fake_engine(self):
        from model_lobotomy.guards import NoopGuard
        from model_lobotomy import reader as reader_mod
        from model_lobotomy.routers import find as find_route

        fake = FakeMuse([
            [call("measure_lift", layer=20, neuron=123, dose=1.0,
                  words=["storm"], prediction=3.0)],
            [call("try_recipe",
                  rows=[{"layer": 20, "neuron": 123, "dose": 1.0}],
                  prediction=["storm"])],
            [call("stop", rows=[{"layer": 20, "neuron": 123,
                                 "dose": 1.0}], words=["storm"],
                  reason="done")],
        ], summaries=["The search used 3 tool calls. The lift was "
                      "4.0. The guard did not block the recipe."])
        events = []
        with mock.patch.object(reader_mod, "steer_lift",
                               return_value={"lift": 4.0}), \
                mock.patch.object(reader_mod, "steer_lift_recipe",
                                  return_value={"lift": 4.0}), \
                mock.patch("model_lobotomy.guards.get_guard",
                           return_value=NoopGuard()), \
                mock.patch.object(find_route.helpers, "_calibrate",
                                  return_value=({}, None)):
            gen = find_route._find_events(
                FakeEng(), fake,
                mock.Mock(goal="bring up storms",
                          prompt="Tell me about your weekend."))
            for chunk in gen:
                events.append(json.loads(chunk))
        stages = [e["stage"] for e in events]
        self.assertEqual(stages, ["prediction", "result", "prediction",
                                  "result", "stop", "final", "summary",
                                  "done"])
        self.assertEqual(events[-1]["result"]["status"], "passed")
        recipe = [e for e in events if e.get("tool") == "try_recipe"
                  and e["stage"] == "result"][0]
        self.assertTrue(recipe["confirmed"])
        self.assertEqual(recipe["result"]["fork"]["edited"], "saw")


if __name__ == "__main__":
    unittest.main()
