"""Unit tests: Create job (stages, Stop, status, reattach) and the
Stop/progress threading through trait_eval (scripted engines only).
"""
from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import create_job  # noqa: E402
import trait_eval as T  # noqa: E402
from trait_store import TraitFamily, TraitStore, UnsupportedTrait  # noqa: E402


class FakeEngine:
    def __init__(self):
        self.spec = SimpleNamespace(layers=4)
        self.revision = "rev123"
        self.lock = threading.Lock()


def validated_store(tmp):
    store = TraitStore(path=Path(tmp) / "traits.json", seed=False)
    store.register(TraitFamily(name="blunt", pos="p", neg="n",
                               judge="j", hand_validated=True,
                               validation={"hand_labels": 12,
                                           "agreement": 0.95}))
    store.register(TraitFamily(name="pirate"))
    return store


def scripted_run(stages=("extract", "tune", "verdict"), out=None):
    def fn(eng, spec, trait, bank, seed, workdir, model_id,
           revision, cancel=None, progress=None, research_mode=False):
        for s in stages:
            if cancel is not None and cancel.is_set():
                raise T.Stopped("stopped")
            if progress is not None:
                progress(s, 1, 1)
        return out or {"verdict": "knob",
                       "frozen": {"layer": 7, "dose": 1.0},
                       "eligibility": {"knob": True},
                       "guard": {"blocked": False}}
    return fn


class CreateRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.jobs = Path(self.tmp.name) / "jobs"
        self.store = validated_store(self.tmp.name)

    def runner(self, **kw):
        kw.setdefault("engine", FakeEngine())
        return create_job.CreateRunner(jobs_dir=self.jobs,
                                       store=self.store, **kw)

    def join(self, runner, job_id):
        runner._jobs[job_id].thread.join(timeout=30)
        self.assertFalse(
            runner._jobs[job_id].thread.is_alive())

    def test_start_runs_stages_to_done(self):
        r = self.runner(run_trait_fn=scripted_run())
        jid = r.start("blunt", seed=0, model_id="m")
        self.join(r, jid)
        st = r.status(jid)
        self.assertEqual(st["status"], "done")
        self.assertEqual(st["stage"], "verdict")
        self.assertEqual(st["verdict"], "knob")
        self.assertEqual(st["frozen"], {"layer": 7, "dose": 1.0})
        self.assertTrue((self.jobs / jid / "job.json").exists())

    def test_stop_sets_stopped(self):
        started = threading.Event()

        def fn(eng, spec, trait, bank, seed, workdir, model_id,
               revision, cancel=None, progress=None,
               research_mode=False):
            started.set()
            cancel.wait(timeout=30)
            raise T.Stopped("stopped")

        r = self.runner(run_trait_fn=fn)
        jid = r.start("blunt", seed=0, model_id="m")
        self.assertTrue(started.wait(timeout=30))
        st = r.stop(jid)
        self.assertEqual(st["status"], "stopped")

    def test_second_job_queues_while_busy(self):
        eng = FakeEngine()
        eng.lock.acquire()
        self.addCleanup(lambda: eng.lock.release()
                        if eng.lock.locked() else None)
        r = self.runner(engine=eng,
                        run_trait_fn=scripted_run())
        jid = r.start("blunt", seed=0, model_id="m")
        self.assertEqual(r.status(jid)["status"], "queued")
        eng.lock.release()
        self.join(r, jid)
        self.assertEqual(r.status(jid)["status"], "done")

    def test_reattach_reads_disk_across_runners(self):
        r = self.runner(run_trait_fn=scripted_run())
        jid = r.start("blunt", seed=0, model_id="m")
        self.join(r, jid)
        r2 = self.runner(run_trait_fn=scripted_run())
        st = r2.reattach(jid)
        self.assertEqual(st["status"], "done")
        self.assertEqual(st["verdict"], "knob")

    def test_unsupported_description_raises(self):
        r = self.runner(run_trait_fn=scripted_run())
        with self.assertRaises(UnsupportedTrait) as ctx:
            r.start("victorian ghost", model_id="m")
        self.assertIn("not supported yet", str(ctx.exception))
        self.assertEqual(r.list_jobs(), [])

    def test_unvalidated_family_raises(self):
        r = self.runner(run_trait_fn=scripted_run())
        with self.assertRaises(UnsupportedTrait) as ctx:
            r.start("pirate", model_id="m")
        self.assertIn("not supported yet", str(ctx.exception))

    def test_failure_records_verbatim(self):
        def boom(*a, **k):
            raise ValueError("boom-xyz-42")
        r = self.runner(run_trait_fn=boom)
        jid = r.start("blunt", seed=0, model_id="m")
        self.join(r, jid)
        st = r.status(jid)
        self.assertEqual(st["status"], "failed")
        self.assertIn("boom-xyz-42", st["error"])

    def test_unknown_job_raises(self):
        r = self.runner(run_trait_fn=scripted_run())
        with self.assertRaises(KeyError):
            r.status("nope")
        with self.assertRaises(KeyError):
            r.reattach("nope")

    def test_list_jobs(self):
        r = self.runner(run_trait_fn=scripted_run())
        a = r.start("blunt", seed=0, model_id="m")
        b = r.start("blunt", seed=1, model_id="m")
        self.join(r, a)
        self.join(r, b)
        self.assertEqual(len(r.list_jobs()), 2)

    def test_warms_cold_engine_before_run(self):
        seen = {}

        class ColdEngine(FakeEngine):
            def __init__(self):
                super().__init__()
                self.model = None
                self.model_id = None

            def load(self, model_id, revision):
                seen["load"] = (model_id, revision)
                self.model = object()
                self.model_id = model_id

        def fn(eng, *a, **k):
            seen["warm"] = eng.model is not None
            return {"verdict": "knob", "frozen": {}, "eligibility": {},
                    "guard": {}}

        r = self.runner(engine=ColdEngine(), run_trait_fn=fn)
        jid = r.start("blunt", seed=0, model_id="m", revision="r")
        self.join(r, jid)
        self.assertEqual(seen["load"], ("m", "r"))
        self.assertTrue(seen["warm"])
        self.assertEqual(r.status(jid)["status"], "done")


class StopThreadingTest(unittest.TestCase):
    def set_cancel(self):
        ev = threading.Event()
        ev.set()
        return ev

    def test_gen_forwards_cancel(self):
        seen = {}

        class Eng:
            def run(self, prompt, edits, medians, max_new_tokens,
                    seed, cancel=None):
                seen["cancel"] = cancel
                return {"text": "t"}

        ev = threading.Event()
        self.assertEqual(T._gen(Eng(), "q", None, cancel=ev), "t")
        self.assertIs(seen["cancel"], ev)

    def test_loops_raise_stopped(self):
        ev = self.set_cancel()
        with self.assertRaises(T.Stopped):
            T.final_gains(None, None, None, {"layer": 0}, None, ["q"],
                          cancel=ev)
        with self.assertRaises(T.Stopped):
            T.delete_gains(None, None, None, {"layer": 0}, ["q"],
                           cancel=ev)
        with self.assertRaises(T.Stopped):
            T.tune_layer_dose(None, None, None, None, {}, {}, ["q"],
                              "j", cancel=ev)
        with self.assertRaises(T.Stopped):
            T.read_gate(None, None, None, {"layer": 0}, None,
                        {"fit": [], "heldout": []}, cancel=ev)
        with self.assertRaises(T.Stopped):
            T.baseline_top1(None, None, {"name": "blunt"}, "j", ["q"],
                            ["q"], cancel=ev)
        with self.assertRaises(T.Stopped):
            T._paired_gains(None, None, None, ["q"], None, +1.0,
                            cancel=ev)
        with self.assertRaises(T.Stopped):
            T.run_trait(None, None, "blunt", [], 0, "/tmp/x", "m",
                        "r", cancel=ev)

    def test_capture_chunked_concats_and_reports(self):
        seen = []

        class Eng:
            def capture(self, texts, layers, point="block_out",
                        reduce="last"):
                return {L: torch.ones(len(texts), 3) * len(texts)
                        for L in layers}

        got = T._capture_chunked(Eng(), [f"t{i}" for i in range(10)],
                                 [0, 1], chunk=4,
                                 progress=lambda s, d, n: seen.append(
                                     (s, d, n)), stage="extract")
        self.assertEqual(tuple(got[0].shape), (10, 3))
        self.assertTrue(torch.equal(got[0][:4],
                                    torch.ones(4, 3) * 4))
        self.assertEqual(seen[-1], ("extract", 10, 10))

    def test_capture_chunked_honors_cancel(self):
        class Eng:
            def capture(self, texts, layers, point="block_out",
                        reduce="last"):
                raise AssertionError("must not run")

        with self.assertRaises(T.Stopped):
            T._capture_chunked(Eng(), ["a"], [0],
                               cancel=self.set_cancel())


if __name__ == "__main__":
    unittest.main()
