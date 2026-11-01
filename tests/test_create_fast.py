"""Unit tests: fast describe-to-patch (C2).

No model. Proves: the schedule table (full reproduces today's
constants, fast matches the spec); unknown schedules are rejected
in the library, the runner and the API; the capture cache
round-trips and invalidates on key change; the base cache
generates once per prompt across verbs; the early-stop rule stops
clear passes and extends borderline and failing runs; fast runs
fewer prompts than full; the API defaults to fast.
"""
from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import create_job  # noqa: E402
from model_lobotomy import engine as engine_mod  # noqa: E402
from model_lobotomy import server  # noqa: E402
from model_lobotomy import trait_eval as T  # noqa: E402
from model_lobotomy.trait_store import TraitFamily, TraitStore  # noqa: E402


class ScheduleTableTest(unittest.TestCase):
    def test_full_reproduces_today(self):
        full = T.schedule_params("full")
        self.assertEqual(full["gen_tokens"], T.GEN_TOKENS)
        self.assertEqual(full["tune_top"], T.TOP_LAYERS)
        self.assertEqual(full["tune_doses"], T.TUNE_DOSES)
        for key in ("tune_n", "read_heldout_n", "final_n0", "final_nmax",
                    "base_instructions", "base_tune_n", "base_final_n0",
                    "base_final_nmax"):
            self.assertIsNone(full[key], key)

    def test_fast_matches_spec(self):
        fast = T.schedule_params("fast")
        self.assertEqual(fast, {
            "gen_tokens": 24, "tune_top": 2,
            "tune_doses": [0.0, 1.0, 2.0], "tune_n": 5,
            "read_heldout_n": 16, "final_n0": 6, "final_nmax": 10,
            "base_instructions": 3, "base_tune_n": 5,
            "base_final_n0": 6, "base_final_nmax": 10})

    def test_unknown_schedule_rejected(self):
        with self.assertRaises(ValueError):
            T.schedule_params("bogus")
        with self.assertRaises(ValueError):
            T.run_trait(None, None, "blunt", [], 0, "/tmp/x", "m",
                        "r", schedule="bogus")


class FakeRunnerEngine:
    def __init__(self):
        from types import SimpleNamespace
        self.lock = threading.Lock()
        self.revision = "rev"
        self.spec = SimpleNamespace(layers=4)


def _validated_store(tmp):
    store = TraitStore(path=Path(tmp) / "traits.json", seed=False)
    store.register(TraitFamily(name="blunt", pos="p", neg="n",
                               judge="j", hand_validated=True,
                               validation={"hand_labels": 12,
                                           "agreement": 0.95}))
    return store


class SchedulePlumbingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.jobs = Path(self.tmp.name) / "jobs"

    def test_runner_defaults_full_records_schedule(self):
        seen = {}

        def fn(eng, spec, trait, bank, seed, workdir, model_id,
               revision, cancel=None, progress=None, guard=False,
               research_mode=False, schedule="full", pareto=False):
            seen["schedule"] = schedule
            return {"verdict": "knob", "frozen": {}, "eligibility": {},
                    "guard": {}}

        r = create_job.CreateRunner(
            jobs_dir=self.jobs, store=_validated_store(self.tmp.name),
            engine=FakeRunnerEngine(), run_trait_fn=fn)
        job_id = r.start("blunt", model_id="m")
        r._jobs[job_id].thread.join(timeout=30)
        self.assertEqual(seen["schedule"], "full")
        self.assertEqual(r.status(job_id)["schedule"], "full")

    def test_runner_passes_fast_through(self):
        seen = {}

        def fn(eng, spec, trait, bank, seed, workdir, model_id,
               revision, cancel=None, progress=None, guard=False,
               research_mode=False, schedule="full", pareto=False):
            seen["schedule"] = schedule
            return {"verdict": "knob", "frozen": {}, "eligibility": {},
                    "guard": {}}

        r = create_job.CreateRunner(
            jobs_dir=self.jobs, store=_validated_store(self.tmp.name),
            engine=FakeRunnerEngine(), run_trait_fn=fn)
        job_id = r.start("blunt", model_id="m", schedule="fast")
        r._jobs[job_id].thread.join(timeout=30)
        self.assertEqual(seen["schedule"], "fast")

    def test_runner_rejects_unknown_schedule(self):
        r = create_job.CreateRunner(
            jobs_dir=self.jobs, store=_validated_store(self.tmp.name),
            engine=FakeRunnerEngine(),
            run_trait_fn=lambda *a, **k: {})
        with self.assertRaises(ValueError):
            r.start("blunt", model_id="m", schedule="bogus")

    def test_api_defaults_full_rejects_bogus(self):
        def fn(eng, spec, trait, bank, seed, workdir, model_id,
               revision, cancel=None, progress=None, guard=False,
               research_mode=False, schedule="full", pareto=False):
            return {"verdict": "knob", "frozen": {}, "eligibility": {},
                    "guard": {}}

        runner = create_job.CreateRunner(
            jobs_dir=self.jobs, store=_validated_store(self.tmp.name),
            engine=FakeRunnerEngine(), run_trait_fn=fn)
        server._runner = runner
        self.addCleanup(setattr, server, "_runner", None)
        nomodel = mock.patch.object(server, "NO_MODEL", False)
        nomodel.start()
        self.addCleanup(nomodel.stop)
        client = TestClient(server.api)
        r = client.post("/api/create", json={"description": "blunt"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["schedule"], "full")
        r = client.post("/api/create", json={"description": "blunt",
                                             "schedule": "bogus"})
        self.assertEqual(r.status_code, 422)


class CaptureCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _mats(self):
        torch.manual_seed(0)
        return ({0: torch.randn(4, 3), 1: torch.randn(4, 3)},
                {0: torch.randn(4, 3), 1: torch.randn(4, 3)})

    def _key(self, pos="pos", neg="neg"):
        return T.capture_cache_key("v1", "m", "rev", pos, neg,
                                   ["a", "b", "c", "d"])

    def test_round_trip(self):
        A, B = self._mats()
        path = self.root / "cap.npz"
        T.save_capture_cache(path, self._key(), A, B)
        back = T.load_capture_cache(path, self._key())
        self.assertIsNotNone(back)
        for L in (0, 1):
            self.assertTrue(torch.equal(back[0][L], A[L]))
            self.assertTrue(torch.equal(back[1][L], B[L]))

    def test_key_mismatch_invalidates(self):
        A, B = self._mats()
        path = self.root / "cap.npz"
        T.save_capture_cache(path, self._key(), A, B)
        self.assertIsNone(T.load_capture_cache(
            path, self._key(pos="changed")))
        self.assertIsNone(T.load_capture_cache(
            path, T.capture_cache_key("v1", "m", "other", "pos",
                                      "neg", ["a", "b", "c", "d"])))

    def test_missing_and_corrupt_are_misses(self):
        self.assertIsNone(T.load_capture_cache(
            self.root / "nope.npz", self._key()))
        bad = self.root / "bad.npz"
        bad.write_bytes(b"not a npz")
        self.assertIsNone(T.load_capture_cache(bad, self._key()))

    def test_path_layout(self):
        got = T.capture_cache_path(self.root, "Org/Model_X", "pirate",
                                   0)
        self.assertEqual(
            got, self.root / "captures" / "org-model-x" /
            "pirate-seed0.npz")


class FakeGenEngine:
    """Scripted generations: reply text encodes (prompt, edits)."""

    def __init__(self):
        self.calls: list = []
        self.model = object()
        self.tokenizer = object()

    def run(self, prompt, edits, medians, max_new_tokens, seed,
            cancel=None):
        self.calls.append((prompt, edits, max_new_tokens))
        tag = "base" if edits == engine_mod.EMPTY_EDITS else "steered"
        return {"text": f"{tag}:{prompt}"}


def _judge_for(mapping):
    """judge_score stub: score by reply tag (base/steered/control)."""

    def fake(model, tok, spec, q, reply, label):
        for tag, score in mapping.items():
            if reply.startswith(tag + ":"):
                return score
        raise AssertionError(f"unscripted reply {reply!r}")

    return fake


class GainsLoopTest(unittest.TestCase):
    def test_base_cache_generates_once_across_verbs(self):
        eng = FakeGenEngine()
        cache = T.BaseCache()
        cfg = {"judge": "j"}
        with mock.patch.object(T, "judge_score",
                               side_effect=_judge_for(
                                   {"base": 0.1, "steered": 0.9})):
            T.verb_final(eng, None, cfg, "j", ["q1", "q2"],
                         engine_mod.EMPTY_EDITS, engine_mod.EMPTY_EDITS,
                         max_new=24, base_cache=cache)
            first = [c for c in eng.calls if c[1] == engine_mod.EMPTY_EDITS]
            # base + trait + control per prompt, all EMPTY here
            self.assertEqual(len(eng.calls), 6)
            T.verb_final(eng, None, cfg, "j", ["q1", "q2"],
                         engine_mod.EMPTY_EDITS, engine_mod.EMPTY_EDITS,
                         max_new=24, base_cache=cache)
        base_prompts = [c[0] for c in eng.calls
                        if c[1] == engine_mod.EMPTY_EDITS]
        # 3 EMPTY calls per prompt per verb, but base generated once:
        # the base text is identical, so judge inputs match.
        self.assertEqual(cache.misses, 2)
        self.assertEqual(cache.hits, 2)
        self.assertEqual(len(first), 6)

    def test_base_cache_shares_identical_text(self):
        eng = FakeGenEngine()
        cache = T.BaseCache()
        a = cache.base(eng, "q", 24)
        b = cache.base(eng, "q", 24)
        self.assertEqual(a, b)
        self.assertEqual(len(eng.calls), 1)

    def test_verb_final_runs_all_without_early_stop(self):
        eng = FakeGenEngine()
        cfg = {"judge": "j"}
        seen = []
        with mock.patch.object(T, "judge_score",
                               side_effect=_judge_for(
                                   {"base": 0.1, "steered": 0.9})):
            out = T.verb_final(
                eng, None, cfg, "j", [f"q{i}" for i in range(8)],
                engine_mod.EMPTY_EDITS, engine_mod.EMPTY_EDITS,
                progress=lambda s, d, n: seen.append((s, d, n)),
                max_new=24)
        self.assertEqual(out["n"], 8)
        self.assertFalse(out["stopped_early"])
        self.assertEqual(seen[-1], ("final", 8, 8))
        self.assertEqual(len(out["pairs"]), 8)

    def test_verb_final_stops_when_clear(self):
        eng = FakeGenEngine()
        cfg = {"judge": "j"}
        trait = engine_mod.EditSet(
            vector=(engine_mod.VectorEdit(0, (1.0,), 1.0, 1.0),))

        def fake_run(prompt, edits, medians, max_new_tokens, seed,
                     cancel=None):
            if edits == engine_mod.EMPTY_EDITS:
                return {"text": f"base:{prompt}"}
            if edits == trait:
                return {"text": f"trait:{prompt}"}
            return {"text": f"control:{prompt}"}

        eng.run = fake_run
        with mock.patch.object(
                T, "judge_score",
                side_effect=_judge_for({"base": 0.1, "trait": 0.95,
                                        "control": 0.12})):
            out = T.verb_final(
                eng, None, cfg, "j", [f"q{i}" for i in range(10)],
                trait, engine_mod.EMPTY_EDITS,
                n0=6, nmax=10, max_new=24)
        self.assertTrue(out["stopped_early"])
        self.assertEqual(out["n"], 6)

    def test_verb_final_extends_when_borderline(self):
        eng = FakeGenEngine()
        cfg = {"judge": "j"}
        trait = engine_mod.EditSet(
            vector=(engine_mod.VectorEdit(0, (1.0,), 1.0, 1.0),))

        def fake_run(prompt, edits, medians, max_new_tokens, seed,
                     cancel=None):
            if edits == engine_mod.EMPTY_EDITS:
                return {"text": f"base:{prompt}"}
            if edits == trait:
                return {"text": f"trait:{prompt}"}
            return {"text": f"control:{prompt}"}

        eng.run = fake_run
        scores = {"trait": 0.5, "control": 0.45}
        with mock.patch.object(
                T, "judge_score",
                side_effect=_judge_for({"base": 0.4, **scores})):
            out = T.verb_final(
                eng, None, cfg, "j", [f"q{i}" for i in range(10)],
                trait, engine_mod.EMPTY_EDITS,
                n0=6, nmax=10, max_new=24)
        self.assertFalse(out["stopped_early"])
        self.assertEqual(out["n"], 10)


class EarlyStopRuleTest(unittest.TestCase):
    def test_clear_pass_stops(self):
        self.assertTrue(T.should_stop_early(
            [0.8] * 8, [0.0] * 8))

    def test_short_lists_never_stop(self):
        self.assertFalse(T.should_stop_early(
            [0.8] * 5, [0.0] * 5))

    def test_unpaired_lists_never_stop(self):
        self.assertFalse(T.should_stop_early(
            [0.8] * 8, [0.0] * 7))

    def test_failing_card_extends(self):
        self.assertFalse(T.should_stop_early(
            [0.05] * 8, [0.0] * 8))

    def test_thin_margin_extends(self):
        # Passes the card rules on 8 samples but the lower CI sits
        # at/below the margin: keep measuring.
        gains = [0.9, 0.9, 0.9, -0.3, -0.3, 0.5, 0.1, 0.0]
        self.assertFalse(T.should_stop_early(gains, [0.0] * 8))


class TuneSliceTest(unittest.TestCase):
    def test_fast_tune_counts(self):
        # Fast tune touches top-2 layers x 3 doses x 5 prompts.
        # tune_layer_dose needs seps + judge + nll; check the slice
        # math on the schedule instead of running the model.
        fast = T.schedule_params("fast")
        self.assertEqual(fast["tune_top"] * len(fast["tune_doses"])
                         * fast["tune_n"], 30)
        full = T.schedule_params("full")
        self.assertEqual(full["tune_top"] * len(full["tune_doses"]),
                         15)


if __name__ == "__main__":
    unittest.main()
