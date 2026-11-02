"""U4 tests: follow-up runner, report enrichment, routes, run import.

Value: protects=the Pareto and repair routes (start, status, stop,
busy, no weights), the report fields the evidence page reads, and the
CLI-run import; fails_when=a follow-up runs twice at once, a status is
lost on reattach, or the report drops pareto/repair/tuning.chosen.
no_claim=real weights (fake job functions and fixtures only).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_lobotomy import create_job  # noqa: E402
from model_lobotomy import followups  # noqa: E402
from model_lobotomy import patch as patch_mod  # noqa: E402
from model_lobotomy import server  # noqa: E402
from model_lobotomy import trait_eval  # noqa: E402

_IMP = importlib.util.spec_from_file_location(
    "import_run", ROOT / "scripts" / "import_run.py")
import_run = importlib.util.module_from_spec(_IMP)
_IMP.loader.exec_module(import_run)

MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def write_run(trait_dir: Path, *, pareto=None, rule=True, guard=None):
    """A finished run_trait work dir with a patch (fixture)."""
    trait_dir.mkdir(parents=True)
    report = {
        "trait": "pirate", "seed": 0, "schedule": "full",
        "fingerprint": {"model_id": MODEL, "revision": "rev"},
        "verdict": "knob", "patch": "knob.lobot", "seconds": 60,
        "frozen": {"layer": 16, "dose": 0.5},
        "tuning": {"table": {"16": {"0.0": {"judge": 0.05, "nll": 0.2,
                                            "fluency": 0.96},
                                    "0.5": {"judge": 0.3, "nll": 0.7,
                                            "fluency": 0.95}}},
                   **({"rule": {"name": "nll+fluency-v1"}} if rule else {})},
        "final": {}, "prompt_baseline": {"top": "Talk like a pirate."},
        "guard": guard or {"blocked": False, "would_block": True,
                           "verified": False,
                           "screens": {"refusal": {"blocked": False,
                                                   "would_block": True},
                                       "privacy": {"blocked": False,
                                                   "would_block": False}}},
    }
    if pareto is not None:
        report["pareto"] = pareto
    (trait_dir / "report.json").write_text(json.dumps(report))
    patch_mod.save_patch(trait_dir / "knob.lobot", direction=torch.ones(8),
                         layer=16, dose=0.5, trait="pirate", recipe={},
                         report_card={}, guard={}, fingerprint={},
                         scale=2.0)
    return report


PARETO = {"verdict": "tradeoff", "points": [
    {"name": "vector@0.5", "kind": "vector", "dose": 0.5,
     "fluency": 0.95, "relevance": 0.9},
    {"name": "prompt", "kind": "prompt", "dose": 0.0, "fluency": 0.9}]}


class FakeEng:
    model = object()
    model_id = MODEL
    revision = "rev"

    def __init__(self):
        self.lock = threading.Lock()


def wait(fn, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        if fn():
            return True
        time.sleep(0.01)
    return False


class FollowupRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "trait"
        write_run(self.dir)
        self.gate = threading.Event()
        self.calls = []

        def fake(eng, workdir, lane, *, cancel=None, progress=None, **kw):
            self.calls.append(kw)
            progress("pareto", 1, 2)
            while not self.gate.is_set():
                trait_eval._check(cancel)
                time.sleep(0.005)
            progress("pareto", 2, 2)
            return {}

        self.runner = followups.FollowupRunner(
            FakeEng(), pareto_fn=fake, repair_fn=fake)

    def test_runs_and_persists(self):
        st = self.runner.start("j", "pareto", self.dir, MODEL)
        self.assertIn(st["status"], ("queued", "running"))
        self.assertTrue(wait(lambda: self.runner.status(
            "j", "pareto", self.dir)["done"] == 1))
        self.gate.set()
        self.assertTrue(wait(lambda: self.runner.status(
            "j", "pareto", self.dir)["status"] == "done"))
        disk = json.loads((self.dir / "pareto.status.json").read_text())
        self.assertEqual((disk["status"], disk["done"], disk["total"]),
                         ("done", 2, 2))

    def test_one_at_a_time(self):
        self.runner.start("j", "pareto", self.dir, MODEL)
        with self.assertRaises(RuntimeError):
            self.runner.start("j", "repair", self.dir, MODEL,
                              method="gradient")
        self.gate.set()

    def test_stop(self):
        self.runner.start("j", "repair", self.dir, MODEL, method="gradient")
        self.assertTrue(wait(lambda: self.runner.status(
            "j", "repair", self.dir)["status"] == "running"))
        self.runner.stop("j", "repair", self.dir)
        self.assertTrue(wait(lambda: self.runner.status(
            "j", "repair", self.dir)["status"] == "stopped"))
        self.assertEqual(self.calls, [{"method": "gradient"}])

    def test_reattach_after_restart_reads_stopped(self):
        (self.dir / "repair.status.json").write_text(json.dumps(
            {"kind": "repair", "status": "running", "done": 1}))
        fresh = followups.FollowupRunner(FakeEng())
        st = fresh.status("j", "repair", self.dir)
        self.assertEqual(st["status"], "stopped")
        self.assertIn("restarted", st["error"])

    def test_needs_a_patch(self):
        (self.dir / "knob.lobot").unlink()
        rep = json.loads((self.dir / "report.json").read_text())
        rep.pop("patch")
        (self.dir / "report.json").write_text(json.dumps(rep))
        with self.assertRaises(ValueError):
            self.runner.start("j", "pareto", self.dir, MODEL)

    def test_unknown_kind_and_method(self):
        with self.assertRaises(ValueError):
            self.runner.start("j", "nope", self.dir, MODEL)
        with self.assertRaises(ValueError):
            followups.repair_job(None, self.dir, method="nope")
        self.assertEqual(followups.repair_stem("project"), "repair")
        self.assertEqual(followups.repair_stem("gradient"),
                         "repair_gradient")


class ReportRoutesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.jobs = Path(self.tmp.name) / "jobs"
        self.job_id = "pirate-fd"
        rep = write_run(self.jobs / self.job_id / "trait")
        (self.jobs / self.job_id / "job.json").write_text(json.dumps(
            import_run.job_record(self.job_id, rep, 1793300000.0)))
        server._runner = create_job.CreateRunner(jobs_dir=self.jobs)
        self.addCleanup(setattr, server, "_runner", None)
        server._followups = followups.FollowupRunner(FakeEng())
        self.addCleanup(setattr, server, "_followups", None)
        self.client = TestClient(server.api)

    def trait_dir(self):
        return self.jobs / self.job_id / "trait"

    def test_report_is_enriched(self):
        (self.trait_dir() / "pareto.json").write_text(json.dumps(PARETO))
        (self.trait_dir() / "repair_gradient.json").write_text(
            json.dumps({"status": "unrepaired"}))
        r = self.client.get(f"/api/traits/{self.job_id}/report").json()
        self.assertEqual(r["pareto"]["verdict"], "tradeoff")
        self.assertEqual(r["repair"], {"gradient": {"status": "unrepaired"}})
        self.assertEqual(r["tuning"]["chosen"],
                         {"layer": 16, "dose": 0.5, "judge": 0.3,
                          "nll": 0.7, "fluency": 0.95})
        self.assertEqual(r["followups"], {"pareto": None, "repair": None})

    def test_list_evidence_summary(self):
        (self.trait_dir() / "pareto.json").write_text(json.dumps(PARETO))
        (item,) = self.client.get("/api/traits").json()["traits"]
        self.assertEqual(item["evidence"], {
            "pareto_verdict": "tradeoff", "dose_rule": "nll+fluency-v1",
            "fluency": 0.95,
            "screens": {"refusal": "would_block", "privacy": "passed"},
            "repair": None})

    def test_start_needs_weights(self):
        with mock.patch.object(server, "REPLAY", True):
            r = self.client.post(f"/api/traits/{self.job_id}/pareto")
        self.assertEqual(r.status_code, 503)

    def test_start_status_and_busy(self):
        gate = threading.Event()

        def fake(eng, workdir, lane, *, cancel=None, progress=None, **kw):
            gate.wait(15)
            (Path(workdir) / "repair_gradient.json").write_text(
                json.dumps({"status": "repaired", **kw}))

        server._followups = followups.FollowupRunner(
            FakeEng(), repair_fn=fake, pareto_fn=fake)
        with mock.patch.object(server, "NO_MODEL", False), \
                mock.patch.object(server, "REPLAY", False):
            r = self.client.post(f"/api/traits/{self.job_id}/repair",
                                 json={"method": "gradient"})
            self.assertEqual(r.status_code, 200, r.text)
            busy = self.client.post(f"/api/traits/{self.job_id}/pareto")
            self.assertEqual(busy.status_code, 409)
            gate.set()
            self.assertTrue(wait(lambda: self.client.get(
                f"/api/traits/{self.job_id}/repair").json()["status"][
                    "status"] == "done"))
        res = self.client.get(f"/api/traits/{self.job_id}/repair").json()
        self.assertEqual(res["result"]["gradient"]["status"], "repaired")
        self.assertEqual(res["result"]["gradient"]["method"], "gradient")

    def test_bad_method_and_unknown_job(self):
        with mock.patch.object(server, "NO_MODEL", False), \
                mock.patch.object(server, "REPLAY", False):
            r = self.client.post(f"/api/traits/{self.job_id}/repair",
                                 json={"method": "nope"})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(self.client.get(
            "/api/traits/nope/pareto").status_code, 404)
        self.assertEqual(self.client.get(
            f"/api/traits/{self.job_id}/repaired/gradient").status_code, 404)

    def test_stop_route(self):
        r = self.client.post(f"/api/traits/{self.job_id}/pareto/stop")
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.json()["status"])


class ImportRunTest(unittest.TestCase):
    def test_import_copies_and_writes_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "run"
            write_run(src, pareto=PARETO)
            dest = import_run.import_run(src, Path(tmp) / "jobs", "p1")
            job = json.loads((dest / "job.json").read_text())
            self.assertEqual((job["status"], job["trait"], job["model_id"],
                              job["frozen"]["dose"], job["pareto"]),
                             ("done", "pirate", MODEL, 0.5, True))
            self.assertTrue((dest / "trait" / "knob.lobot").is_file())
            with self.assertRaises(FileExistsError):
                import_run.import_run(src, Path(tmp) / "jobs", "p1")


class CreateParetoOptionTest(unittest.TestCase):
    def test_default_follows_schedule_and_reaches_run_trait(self):
        seen = []

        def fake_run_trait(*a, **kw):
            seen.append(kw.get("pareto"))
            return {"verdict": "no_knob"}

        with tempfile.TemporaryDirectory() as tmp:
            eng = FakeEng()
            eng.spec = None
            runner = create_job.CreateRunner(
                jobs_dir=Path(tmp), engine=eng, run_trait_fn=fake_run_trait)
            for sched, opt in (("full", None), ("fast", None),
                               ("fast", True)):
                jid = runner.start("pirate", model_id=MODEL,
                                   schedule=sched, pareto=opt)
                self.assertTrue(wait(lambda: runner.status(jid)["status"]
                                     == "done"))
        self.assertEqual(seen, [True, None, True])
        self.assertIn("pareto", create_job.STAGES)
        self.assertIn("guard", create_job.STAGES)


if __name__ == "__main__":
    unittest.main()
