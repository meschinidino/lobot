"""Muse Spark helpers and the Create agent: scripted fakes, no API, no model."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_lobotomy import agent as agent_mod  # noqa: E402
from model_lobotomy import drafts  # noqa: E402
from model_lobotomy import muse as muse_mod  # noqa: E402
from model_lobotomy import trait_eval  # noqa: E402
from model_lobotomy.trait_store import TraitStore  # noqa: E402

PARTS = {"pos": "Answer like an old-west cowboy.",
         "neg": "Answer plainly.",
         "judge": "talks like an old-west cowboy",
         "baselines": [f"b{i}" for i in range(6)]}


class Scripted:
    """complete(prompt): answers by prompt kind; records every prompt."""

    def __init__(self, drafts_=None, decide=None, labels=None):
        self.prompts = []
        self.drafts = list(drafts_ or [json.dumps(PARTS)])
        self.decide = list(decide or [])
        self.labels = labels
        self.model = "fake-muse"
        self.usage = {"calls": 0}

    def __call__(self, prompt):
        self.prompts.append(prompt)
        self.usage["calls"] += 1
        if prompt.startswith("You design"):
            return self.drafts.pop(0) if len(self.drafts) > 1 \
                else self.drafts[0]
        if prompt.startswith("Behavior:"):
            ids = [ln[1:-1] for ln in prompt.splitlines()
                   if ln.startswith("[r")]
            return json.dumps(self.labels or {i: int(i[1:]) % 2 == 0
                                              for i in ids})
        if prompt.startswith("Explain"):
            return "It worked. The cowboy voice rose."
        if prompt.startswith("You run"):
            return self.decide.pop(0)
        raise AssertionError(prompt[:40])


class MuseTest(unittest.TestCase):
    def test_draft_retries_with_feedback(self):
        bad = json.dumps({**PARTS, "judge": "The reply should rhyme."})
        fake = Scripted(drafts_=[bad, json.dumps(PARTS)])
        got = muse_mod.Muse(fake).draft_parts("talks like a cowboy")
        self.assertEqual(got["judge"], "talks like an old-west cowboy")
        self.assertIn("rejected", fake.prompts[1])

    def test_draft_gives_up(self):
        bad = json.dumps({**PARTS, "judge": "The reply should rhyme."})
        with self.assertRaises(muse_mod.MuseError):
            muse_mod.Muse(Scripted(drafts_=[bad])).draft_parts("x")

    def test_prelabel_keeps_known_ids_and_bools(self):
        fake = Scripted(labels={"r00": True, "r01": "yes", "zz": True})
        got = muse_mod.Muse(fake).prelabel(
            "d", [{"reply_id": "r00", "reply": "a"},
                  {"reply_id": "r01", "reply": "b"}])
        self.assertEqual(got, {"r00": True})

    def test_decide_validates_action(self):
        m = muse_mod.Muse(Scripted(decide=['{"action": "explode"}']))
        with self.assertRaises(muse_mod.MuseError):
            m.decide("d", "validate", "s", agent_mod.ACTIONS)
        m = muse_mod.Muse(Scripted(decide=['{"action": "redraft"}']))
        with self.assertRaises(muse_mod.MuseError):
            m.decide("d", "validate", "s", agent_mod.ACTIONS)

    def test_explain_sends_only_summary_numbers(self):
        fake = Scripted()
        report = {"trait": "t", "verdict": "knob", "secret": "no",
                  "frozen": {"layer": 17, "dose": 0.5,
                             "tuning_gain": 0.14567}}
        muse_mod.Muse(fake).explain(report)
        self.assertNotIn("secret", fake.prompts[0])
        self.assertIn("0.146", fake.prompts[0])

    def test_missing_key_is_muse_error(self):
        with mock.patch.dict("os.environ", {"META_API_KEY": ""}):
            with self.assertRaises(muse_mod.MuseError):
                muse_mod.ApiCompleter()


class FakeEng:
    model = object()
    tokenizer = object()
    spec = None
    model_id = "meta-llama/x"
    revision = "rev"


def _replies(*a, **k):
    return [{"reply_id": f"r{i:02d}", "prompt": f"q{i}",
             "reply": f"{'cowboy' if i % 2 == 0 else 'plain'} {i}",
             "side": "pos" if i % 2 == 0 else "neg"} for i in range(20)]


def _judge(model, tok, spec, q, reply, label):
    return 0.9 if reply.startswith("cowboy") else 0.1


class AgentTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.t = Path(tmp.name)
        self.store = TraitStore(path=self.t / "traits.json") \
            if "path" in TraitStore.__init__.__code__.co_varnames \
            else TraitStore()
        self.store.save = lambda: None
        self.runs = []
        for p in (mock.patch.object(drafts, "candidate_replies", _replies),
                  mock.patch.object(trait_eval, "judge_score", _judge)):
            p.start()
            self.addCleanup(p.stop)

    def _agent(self, fake, verdicts=("knob",), **kw):
        verdicts = list(verdicts)

        def run_trait(eng, spec, name, bank, seed, workdir, *a, **k):
            Path(workdir).mkdir(parents=True, exist_ok=True)
            self.runs.append(name)
            v = verdicts.pop(0)
            return {"verdict": v, "trait": name, "seconds": 1.0,
                    "frozen": {"layer": 17, "dose": 0.5}
                    if v == "knob" else None,
                    "eligibility": {"read": v == "knob"}}

        return agent_mod.CreateAgent(
            FakeEng(), muse_mod.Muse(fake), store=self.store,
            workdir_root=self.t / "agents", run_trait_fn=run_trait,
            audit_log=self.t / "audit.jsonl", **kw)

    def test_happy_path_registers_runs_explains(self):
        ag = self._agent(Scripted())
        out = ag.run("talks like an old-west cowboy")
        self.assertEqual(out["status"], "done")
        self.assertEqual(out["explanation"],
                         "It worked. The cowboy voice rose.")
        self.assertEqual([s["step"] for s in out["steps"]],
                         ["draft", "labels", "validate", "register",
                          "create", "explain"])
        fam = self.store.get(out["family"])
        self.assertEqual(fam.judge, "talks like an old-west cowboy")
        events = [json.loads(l)["event"] for l in
                  (self.t / "audit.jsonl").read_text().splitlines()]
        self.assertEqual(events[0], "agent.draft")
        self.assertTrue((self.t / "agents" / ag.id / "agent.json").is_file())

    def test_failed_validation_redrafts_then_passes(self):
        wrong = {f"r{i:02d}": i % 2 == 1 for i in range(20)}
        fake = Scripted(decide=[json.dumps({
            "action": "redraft", "reason": "labels disagree",
            "description": "speaks in a cowboy voice"})])
        calls = {"n": 0}
        real = fake.__call__

        def labels_first_wrong(prompt):
            if prompt.startswith("Behavior:"):
                calls["n"] += 1
                if calls["n"] == 1:
                    return json.dumps(wrong)
            return real(prompt)

        m = muse_mod.Muse(labels_first_wrong)
        ag = agent_mod.CreateAgent(
            FakeEng(), m, store=self.store,
            workdir_root=self.t / "agents",
            run_trait_fn=lambda *a, **k: (Path(a[5]).mkdir(
                parents=True, exist_ok=True) or {"verdict": "knob",
                                                  "frozen": {}}),
            audit_log=self.t / "audit.jsonl")
        out = ag.run("talks like an old-west cowboy")
        self.assertEqual(out["status"], "done")
        self.assertEqual(out["rounds"], 2)
        decide = [s for s in out["steps"] if s["step"] == "decide"][0]
        self.assertEqual(decide["action"], "redraft")
        # Round 2 drafts from Muse's rewrite; labels keep the user's words.
        drafts_ = [p for p in fake.prompts if p.startswith("You design")]
        self.assertIn("speaks in a cowboy voice", drafts_[1])
        labels_ = [p for p in fake.prompts if p.startswith("Behavior:")]
        self.assertTrue(all("talks like an old-west cowboy" in p
                            for p in labels_))

    def test_no_knob_then_stop(self):
        fake = Scripted(decide=['{"action": "stop", "reason": "no dose"}'])
        out = self._agent(fake, verdicts=("no_knob",)).run("talks slowly")
        self.assertEqual(out["status"], "stopped")
        self.assertIn("no coherent dose", out["why"])

    def test_budget_ends_without_asking_muse(self):
        out = self._agent(Scripted(), verdicts=("no_knob",),
                          max_rounds=1).run("talks slowly")
        self.assertEqual(out["status"], "stopped")
        last = out["steps"][-1]
        self.assertEqual((last["step"], last["by"]), ("decide", "harness"))

    def test_refused_description_never_calls_muse(self):
        fake = Scripted()
        with mock.patch.object(agent_mod.guard_mod,
                               "draft_description_allowed",
                               return_value=(False, "Not allowed.")):
            out = self._agent(fake).run("anything")
        self.assertEqual(out["status"], "refused")
        self.assertEqual(fake.prompts, [])

    def test_user_confirm_overrides_labels(self):
        seen = {}

        def confirm(replies, labels):
            seen["n"] = len(labels)
            return {r["reply_id"]: r["side"] == "pos" for r in replies}

        out = self._agent(Scripted(labels={"r00": True}),
                          confirm_labels=confirm).run("talks like a cowboy")
        self.assertEqual(out["status"], "done")
        self.assertEqual(seen["n"], 1)
        lab = [s for s in out["steps"] if s["step"] == "labels"][0]
        self.assertEqual(lab["by"], "user")


class RoutesTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from model_lobotomy import create_job, server
        from tests.test_traits_api import _write_job
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        jobs = Path(tmp.name) / "jobs"
        jobs.mkdir()
        server._runner = create_job.CreateRunner(jobs_dir=jobs)
        self.addCleanup(setattr, server, "_runner", None)
        self.job = "pirate-0-20260101-000000-000"
        _write_job(jobs, self.job, trait="pirate")
        self.jobs, self.server = jobs, server
        self.client = TestClient(server.api)

    def test_explain_calls_muse_once_then_caches(self):
        fake = Scripted()
        with mock.patch.object(muse_mod, "ApiCompleter",
                               return_value=fake):
            a = self.client.post(f"/api/traits/{self.job}/explain")
            b = self.client.post(f"/api/traits/{self.job}/explain")
        self.assertEqual(a.status_code, 200, a.text)
        self.assertFalse(a.json()["cached"])
        self.assertTrue(b.json()["cached"])
        self.assertEqual(len(fake.prompts), 1)

    def test_explain_refreshes_when_the_numbers_change(self):
        fake = Scripted()
        with mock.patch.object(muse_mod, "ApiCompleter",
                               return_value=fake):
            self.client.post(f"/api/traits/{self.job}/explain")
            (self.jobs / self.job / "trait" / "pareto.json").write_text(
                json.dumps({"verdict": "tradeoff", "verdict_prompt": "prompt",
                            "base": {"trait": 0.1}, "points": [
                                {"name": "prompt", "kind": "prompt",
                                 "dose": 0.0, "trait": 0.9, "cost": 0.2,
                                 "relevance": 0.7, "fluency": 0.9}]}))
            c = self.client.post(f"/api/traits/{self.job}/explain")
        self.assertFalse(c.json()["cached"])
        self.assertEqual(len(fake.prompts), 2)
        self.assertIn("prompt_comparison", fake.prompts[1])
        self.assertIn("tradeoff", fake.prompts[1])

    def test_no_key_is_503_with_hint(self):
        with mock.patch.dict("os.environ", {"META_API_KEY": ""}):
            r = self.client.post(f"/api/traits/{self.job}/explain")
        self.assertEqual(r.status_code, 503)
        self.assertIn("META_API_KEY", r.json()["detail"])

    def test_agent_refuses_harmful_description(self):
        with mock.patch.object(self.server.guard_mod,
                               "draft_description_allowed",
                               return_value=(False, "Not allowed.")):
            r = self.client.post("/api/agent",
                                 json={"description": "x"})
        self.assertEqual(r.status_code, 422)

    def test_agent_unknown_run_404(self):
        self.assertEqual(self.client.get("/api/agent/nope").status_code,
                         404)


if __name__ == "__main__":
    unittest.main()
