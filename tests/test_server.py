"""Unit tests: FastAPI server stream contract (server.py).

Value: protects=NDJSON stage order, split char offsets, lock release,
Stop, input errors; fails_when=the frontend contract drifts or the run lock
leaks after an error. no_claim=real generation (the engine is stubbed
here; the model-backed path stays in the opt-in integration lane).
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app  # noqa: E402
import create_job  # noqa: E402
import guard as guard_mod  # noqa: E402
import server  # noqa: E402
import steer  # noqa: E402
from trait_store import TraitFamily, TraitStore  # noqa: E402

BASE_WORDS = "I am Qwen , a model".split()
STEER_WORDS = "I am an AI that wants".split()
VOCAB = BASE_WORDS + STEER_WORDS


class FakeTokenizer:
    """Token id i decodes to VOCAB[i]; words join with single spaces."""

    def __init__(self):
        self.wrapped_chats: list = []

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(VOCAB[i] for i in ids)

    def encode(self, s, add_special_tokens=False):
        return [0]

    def __call__(self, text, **kw):
        return {"input_ids": [[0, 1, 2, 3, 4]]}

    def apply_chat_template(self, messages, **kw):
        self.wrapped_chats.append(messages)
        return "CHAT:" + "|".join(m["content"] for m in messages)


def _gen(words, hook_calls=3):
    ids = [VOCAB.index(w) if w in BASE_WORDS else len(BASE_WORDS) + STEER_WORDS.index(w)
           for w in words]
    return {"text": " ".join(words), "new_tokens": len(ids), "token_ids": ids,
            "seconds": 0.1, "prefill_s": 0.01, "decode_s": 0.09, "tok_per_s": 50.0,
            "peak_mib": 1.0, "driver_mib": 1.0, "hook_calls": hook_calls,
            "reads": {}, "engine_seconds": 0.1}


ROW = {"layer": 14, "neuron": 8949, "dose": 1.0}


class FakeEngine:
    """Stub for server.get_engine: warm, canned runs, real lock."""

    def __init__(self, outputs):
        self.lock = threading.Lock()
        self.model = object()
        self.tokenizer = FakeTokenizer()
        self.model_id = "Qwen/Qwen3.5-4B"
        self.revision = "stub-rev"
        self.outputs = list(outputs)
        self.runs = []
        self.loads = []
        self.guard_block = False
        self.stream_script = []
        self.stream_script_single = []
        self.capture_script = {}
        self.capture_calls: list = []
        self.spec = None

    def load(self, model_id):
        self.loads.append(model_id)
        self.model_id = model_id
        self.model = object()
        return 1

    @contextmanager
    def clean(self):
        yield

    def run(self, prompt, edits, **kw):
        self.runs.append((prompt, edits, kw))
        if prompt in guard_mod.REFUSAL_SET:
            return {"text": "Sorry, I can't help with that."}
        return self.outputs.pop(0)

    def fingerprint(self):
        return {"rev": "stub-rev"}

    def prefill(self, prompt, edits=None, medians=None):
        if getattr(self, "guard_block", False) and edits is not None:
            return torch.tensor([0.0, 0.0, 0.0, 10.0])
        return torch.tensor([10.0, 0.0, 0.0, 0.0])

    def run_batched_stream(self, prompt, edit_sets, **kw):
        self.runs.append((prompt, edit_sets, kw))
        for ev in self.stream_script:
            yield ev

    def run_stream(self, prompt, edits, **kw):
        self.runs.append((prompt, edits, kw))
        for ev in self.stream_script_single:
            yield ev

    def capture(self, texts, layers, point="mlp_in", reduce="mean_1"):
        self.capture_calls.append((texts, layers, point, reduce))
        return self.capture_script

    def capture_with_edits(self, texts, layers, edits, medians,
                           point="mlp_in", reduce="none"):
        self.capture_calls.append((texts, layers, point, reduce))
        self.runs.append(("capture", edits, {}))
        return self.capture_script


class ServerStreamTest(unittest.TestCase):
    def setUp(self):
        self.eng = FakeEngine([_gen(BASE_WORDS), _gen(STEER_WORDS)])
        self.patches = [
            mock.patch.object(server, "NO_MODEL", False),
            mock.patch.object(server, "get_engine", return_value=self.eng),
            mock.patch.object(steer, "calibration_cache_key", return_value={}),
            mock.patch.object(steer, "calibrate_layer_medians",
                              return_value=({}, {"cache_hit": True,
                                                 "key": {"prompts_sha256": "abc123"}})),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(server.api)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        server._cancel.clear()

    def _events(self, body):
        r = self.client.post("/api/run", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        return [json.loads(line) for line in r.text.splitlines() if line]

    def test_stage_order_and_split_offsets(self):
        ev = self._events({"prompt": "Who are you?", "rows": [ROW]})
        self.assertEqual([e["stage"] for e in ev],
                         ["weights", "guard", "baseline", "steered", "complete"])
        res = ev[-1]["result"]
        self.assertEqual(res["split"]["token"], 2)
        # The offset is the start of the divergent token, leading space included
        # (BPE tokens carry it: " an"). The frontend skips it before marking.
        self.assertEqual(res["split"]["base_char"], 4)
        self.assertEqual(res["split"]["steered_char"], 4)
        self.assertTrue(res["baseline"]["text"][4:].startswith(" Qwen"))
        self.assertTrue(res["steered"]["text"][4:].startswith(" an"))
        self.assertLessEqual(len(res["post"]), 280)
        self.assertIn("L14N8949@1", res["recipe"])
        self.assertFalse(self.eng.lock.locked())

    def test_hooks_not_fired_is_error_and_releases_lock(self):
        self.eng.outputs = [_gen(BASE_WORDS), _gen(STEER_WORDS, hook_calls=0)]
        ev = self._events({"prompt": "Who are you?", "rows": [ROW]})
        self.assertEqual(ev[-1]["stage"], "error")
        self.assertIn("did not fire", ev[-1]["message"])
        self.assertFalse(self.eng.lock.locked())

    def test_engine_exception_is_error_and_releases_lock(self):
        self.eng.model = None
        self.eng.load = mock.Mock(side_effect=RuntimeError("disk full"))
        ev = self._events({"prompt": "Who are you?", "rows": [ROW]})
        self.assertEqual(ev[-1], {"stage": "error", "message": "Run failed: disk full"})
        self.assertFalse(self.eng.lock.locked())

    def test_busy_lock_returns_409(self):
        self.eng.lock.acquire()
        try:
            r = self.client.post("/api/run", json={"prompt": "x", "rows": [ROW]})
        finally:
            self.eng.lock.release()
        self.assertEqual(r.status_code, 409)

    def test_input_errors(self):
        cases = [
            ({"prompt": "  ", "rows": [ROW]}, 400),
            ({"prompt": "x", "rows": []}, 400),
            ({"prompt": "x", "rows": [{**ROW, "layer": 99}]}, 400),
            ({"prompt": "x", "rows": [{**ROW, "dose": 5}]}, 422),
            ({"prompt": "x", "rows": [ROW] * (app.MAX_ROWS + 1)}, 400),
        ]
        for body, code in cases:
            with self.subTest(body=body):
                self.assertEqual(self.client.post("/api/run", json=body).status_code, code)

    def test_stop_during_baseline_skips_steered_and_releases_lock(self):
        real_run = self.eng.run

        def gen(prompt, edits, **k):
            if prompt not in guard_mod.REFUSAL_SET:
                self.assertIs(k["cancel"], server._cancel)
                server._cancel.set()  # the user presses Stop mid-baseline
            return real_run(prompt, edits, **k)

        with mock.patch.object(self.eng, "run", side_effect=gen) as g:
            ev = self._events({"prompt": "Who are you?", "rows": [ROW]})
        self.assertEqual(ev[-1], {"stage": "stopped", "after": "baseline"})
        self.assertNotIn("steered", [e["stage"] for e in ev])
        # 16 guard verification runs (8 prompts x natural/steered) + baseline
        self.assertEqual(g.call_count, 1 + 2 * len(guard_mod.REFUSAL_SET))
        self.assertFalse(self.eng.lock.locked())

    def test_new_run_clears_an_old_stop(self):
        server._cancel.set()  # stale Stop from an earlier, finished run
        ev = self._events({"prompt": "Who are you?", "rows": [ROW]})
        self.assertEqual(ev[-1]["stage"], "complete")

    def test_stop_before_stream_starts_stops_that_run(self):
        # A Stop can land after /api/run returns but before the stream runs.
        r = self.client.post("/api/run", json={"prompt": "x", "rows": [ROW]})
        self.assertEqual(r.status_code, 200)
        server._cancel.set()
        gen = server._run_stream("x", [steer.RecipeRow(**ROW)], app.active_spec())
        events = [json.loads(b) for b in gen]
        self.assertEqual(events[-1], {"stage": "stopped", "after": "weights"})
        self.assertFalse(self.eng.lock.locked())

    def test_stop_when_idle_reports_not_stopping(self):
        self.assertEqual(self.client.post("/api/stop").json(), {"stopping": False})
        server._cancel.clear()

    def test_no_model_mode_returns_503(self):
        with mock.patch.object(server, "NO_MODEL", True):
            r = self.client.post("/api/run", json={"prompt": "x", "rows": [ROW]})
        self.assertEqual(r.status_code, 503)


class CancelCriteriaTest(unittest.TestCase):
    def test_flags_every_row_once_event_is_set(self):
        ev = threading.Event()
        crit = steer.CancelCriteria(ev)
        ids = torch.zeros((2, 5), dtype=torch.long)
        self.assertEqual(crit(ids, None).tolist(), [False, False])
        ev.set()
        self.assertEqual(crit(ids, None).tolist(), [True, True])


class ServerStaticTest(unittest.TestCase):
    def test_index_and_example_served(self):
        c = TestClient(server.api)
        self.assertIn("lobot", c.get("/").text)
        ex = c.get("/example.json").json()
        self.assertEqual(ex["source"], "artifacts/v1-20261029-042822")
        self.assertEqual(ex["split"]["token"], 2)


class SameOriginTest(unittest.TestCase):
    """SameOriginOnly (D5): cross-origin and DNS-rebound POSTs -> 403; same-origin loopback and no-Origin pass."""

    def setUp(self):
        self.client = TestClient(server.api, base_url="http://127.0.0.1:7860")
        self._active = app._active_model_id

    def tearDown(self):
        app._active_model_id = self._active

    def _switch(self, headers=None, client=None):
        return (client or self.client).post("/api/model", json={"model_id": "Qwen/Qwen3.5-4B"},
                                            headers=headers or {})

    def test_cross_origin_rejected(self):
        r = self._switch({"Origin": "http://evil.test"})
        self.assertEqual(r.status_code, 403, r.text)
        self.assertIn("Cross-origin", r.json()["detail"])

    def test_same_origin_allowed(self):
        r = self._switch({"Origin": "http://127.0.0.1:7860"})
        self.assertEqual(r.status_code, 200, r.text)

    def test_ipv6_loopback_allowed(self):
        c = TestClient(server.api, base_url="http://[::1]:7860")
        self.assertEqual(self._switch({"Origin": "http://[::1]:7860"}, c).status_code, 200)

    def test_dns_rebinding_rejected(self):
        # A hostile domain re-pointed at 127.0.0.1: Origin and Host match each other, but not loopback.
        c = TestClient(server.api, base_url="http://evil.example:7860")
        r = self._switch({"Origin": "http://evil.example:7860"}, c)
        self.assertEqual(r.status_code, 403, r.text)

    def test_missing_origin_allowed(self):
        r = self._switch()
        self.assertEqual(r.status_code, 200, r.text)

    def test_hostname_parsing(self):
        self.assertEqual(server._hostname("127.0.0.1:7860"), "127.0.0.1")
        self.assertEqual(server._hostname("[::1]:7860"), "::1")
        self.assertEqual(server._hostname("LocalHost"), "localhost")


class ReplayModeTest(unittest.TestCase):
    """Replay mode answers recorded setups only, with the run contract and no weights."""

    def setUp(self):
        self.ex = json.loads((ROOT / "web" / "example.json").read_text())
        self.patches = [mock.patch.object(server, "REPLAY", True),
                        mock.patch.object(server, "REPLAY_STAGE_S", 0.0),
                        mock.patch.object(server, "_run_stream", side_effect=AssertionError("model path used")),
                        mock.patch.object(app, "_active_model_id", self.ex["model_id"])]
        for p in self.patches:
            p.start()
        self.client = TestClient(server.api)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        server._cancel.clear()

    def test_recorded_setup_replays_with_the_run_contract(self):
        rows = list(reversed(self.ex["rows"]))  # row order does not matter
        r = self.client.post("/api/run", json={"prompt": self.ex["prompt"] + " ", "rows": rows})
        self.assertEqual(r.status_code, 200, r.text)
        ev = [json.loads(line) for line in r.text.splitlines() if line]
        self.assertEqual([e["stage"] for e in ev], ["weights", "baseline", "steered", "complete"])
        res = ev[-1]["result"]
        self.assertTrue(res["replay"])
        self.assertEqual(res["steered"]["text"], self.ex["steered"]["text"])
        self.assertEqual(res["split"], self.ex["split"])
        self.assertFalse(server.get_engine().lock.locked())

    def test_dose_zero_replays_the_baseline(self):
        rows = [{**self.ex["rows"][0], "dose": 0}]
        r = self.client.post("/api/run", json={"prompt": self.ex["prompt"], "rows": rows})
        self.assertEqual(r.status_code, 200, r.text)
        res = [json.loads(line) for line in r.text.splitlines() if line][-1]["result"]
        self.assertEqual(res["steered"]["text"], self.ex["baseline"]["text"])
        self.assertIsNone(res["split"])
        self.assertEqual(res["rows"], rows)

    def test_unrecorded_setup_is_refused(self):
        rows = [{**self.ex["rows"][0], "dose": 0.5}]
        r = self.client.post("/api/run", json={"prompt": self.ex["prompt"], "rows": rows})
        self.assertEqual(r.status_code, 404)
        self.assertIn("not recorded", r.json()["detail"])

    def test_stop_during_replay_stops_and_releases_lock(self):
        server._cancel.set()
        events = [json.loads(b) for b in server._replay_stream(server._recordings()[0])]
        self.assertEqual(events[-1], {"stage": "stopped", "after": "weights"})
        self.assertFalse(server.get_engine().lock.locked())

    def test_every_sweep_file_is_replayable(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            web = Path(d)
            (web / "example.json").write_text(json.dumps(self.ex))
            row = {"layer": 11, "neuron": 1343, "dose": 0.5}
            sweep = {"model_id": "meta-llama/Llama-3.2-3B-Instruct", "prompt": "p", "baseline": {"text": "a b", "tokens": 2},
                     "runs": [{"rows": [row], "text": "a c", "tokens": 2, "split": None, "seconds": 1.0}]}
            (web / "sweep.llama-32-3b-instruct.json").write_text(json.dumps(sweep))
            with mock.patch.object(server, "WEB", web):
                self.assertEqual(len(server._recordings()), 2)
                rec = server._find_recording("meta-llama/Llama-3.2-3B-Instruct", "p", [row])
        self.assertEqual(rec["steered"]["text"], "a c")

    def test_state_reports_replay(self):
        self.assertTrue(self.client.get("/api/state").json()["replay"])


def _validated_store(tmp):
    store = TraitStore(path=Path(tmp) / "traits.json", seed=False)
    store.register(TraitFamily(name="blunt", pos="p", neg="n",
                               judge="j", hand_validated=True,
                               validation={"hand_labels": 12,
                                           "agreement": 0.95}))
    return store


def _scripted_trait(out=None):
    def fn(eng, spec, trait, bank, seed, workdir, model_id,
           revision, cancel=None, progress=None,
           guard=False, research_mode=False):
        if progress is not None:
            progress("tune", 1, 1)
        return out or {"verdict": "knob",
                       "frozen": {"layer": 7, "dose": 1.0},
                       "eligibility": {"knob": True},
                       "guard": {"blocked": False}}
    return fn


class GuardStageTest(unittest.TestCase):
    def setUp(self):
        self.eng = FakeEngine([_gen(BASE_WORDS), _gen(STEER_WORDS)])
        self.patches = [
            mock.patch.object(server, "NO_MODEL", False),
            mock.patch.object(server, "get_engine", return_value=self.eng),
            mock.patch.object(steer, "calibration_cache_key", return_value={}),
            mock.patch.object(steer, "calibrate_layer_medians",
                              return_value=({}, {"cache_hit": True,
                                                 "key": {"prompts_sha256": "abc123"}})),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(server.api)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        server._cancel.clear()

    def _events(self, body):
        r = self.client.post("/api/run", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        return [json.loads(line) for line in r.text.splitlines() if line]

    def test_blocked_run_stops_after_guard(self):
        self.eng.guard_block = True
        ev = self._events({"prompt": "Who are you?", "rows": [ROW],
                           "guard": True})
        self.assertEqual([e["stage"] for e in ev],
                         ["weights", "guard", "blocked"])
        self.assertTrue(ev[-1]["guard"]["blocked"])
        self.assertTrue(ev[-1]["guard"]["enforced"])
        self.assertEqual(self.eng.runs, [])
        self.assertFalse(self.eng.lock.locked())

    def test_research_mode_passes_blocking_edit(self):
        self.eng.guard_block = True
        ev = self._events({"prompt": "Who are you?", "rows": [ROW],
                           "guard": True, "research_mode": True})
        self.assertEqual(ev[-1]["stage"], "complete")

    def test_default_run_records_without_blocking(self):
        self.eng.guard_block = True
        ev = self._events({"prompt": "Who are you?", "rows": [ROW]})
        self.assertEqual(ev[-1]["stage"], "complete")
        g = ev[-1]["result"]["guard"]
        self.assertFalse(g["blocked"])
        self.assertTrue(g["would_block"])
        self.assertFalse(g["enforced"])


class FamiliesApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        server._runner = create_job.CreateRunner(
            jobs_dir=Path(self.tmp.name) / "jobs",
            store=_validated_store(self.tmp.name),
            engine=FakeEngine([]))
        self.addCleanup(setattr, server, "_runner", None)
        self.client = TestClient(server.api)

    def test_list_families(self):
        fams = self.client.get("/api/families").json()["families"]
        self.assertEqual([(f["name"], f["hand_validated"]) for f in fams],
                         [("blunt", True)])


class CreateApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runner = create_job.CreateRunner(
            jobs_dir=Path(self.tmp.name) / "jobs",
            store=_validated_store(self.tmp.name),
            engine=FakeEngine([]),
            run_trait_fn=_scripted_trait())
        server._runner = self.runner
        self.addCleanup(setattr, server, "_runner", None)
        self.no_model = mock.patch.object(server, "NO_MODEL", False)
        self.no_model.start()
        self.addCleanup(self.no_model.stop)
        self.client = TestClient(server.api)

    def _join(self, job_id):
        self.runner._jobs[job_id].thread.join(timeout=30)

    def test_create_and_status(self):
        r = self.client.post("/api/create",
                             json={"description": "blunt", "seed": 0})
        self.assertEqual(r.status_code, 200, r.text)
        job_id = r.json()["job_id"]
        self._join(job_id)
        st = self.client.get(f"/api/create/{job_id}").json()
        self.assertEqual(st["status"], "done")
        self.assertEqual(st["verdict"], "knob")
        self.assertEqual(st["trait"], "blunt")

    def test_unsupported_description_is_422(self):
        r = self.client.post("/api/create",
                             json={"description": "victorian ghost"})
        self.assertEqual(r.status_code, 422)
        self.assertIn("not supported yet", r.json()["detail"])

    def test_unknown_job_is_404(self):
        self.assertEqual(
            self.client.get("/api/create/nope").status_code, 404)
        self.assertEqual(
            self.client.post("/api/create/nope/stop").status_code, 404)

    def test_stop_job(self):
        started = threading.Event()

        def fn(eng, spec, trait, bank, seed, workdir, model_id,
               revision, cancel=None, progress=None,
               guard=False, research_mode=False):
            started.set()
            cancel.wait(timeout=30)
            import trait_eval
            raise trait_eval.Stopped("stopped")

        self.runner._run_trait = fn
        job_id = self.client.post(
            "/api/create", json={"description": "blunt"}).json()["job_id"]
        self.assertTrue(started.wait(timeout=30))
        st = self.client.post(f"/api/create/{job_id}/stop").json()
        self.assertEqual(st["status"], "stopped")

    def test_jobs_list(self):
        r = self.client.post("/api/create",
                             json={"description": "blunt"})
        self._join(r.json()["job_id"])
        jobs = self.client.get("/api/jobs").json()["jobs"]
        self.assertEqual(len(jobs), 1)


class UpdateApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.eng = FakeEngine([])
        self.eng.stream_script = [
            {"type": "token", "col": 0, "text": "Hello"},
            {"type": "token", "col": 1, "text": "Hi"},
            {"type": "done", "result": {
                "rows": [{"text": "Hello", "new_tokens": 1,
                          "token_ids": [1]},
                         {"text": "Hi", "new_tokens": 1,
                          "token_ids": [2]}],
                "fallback_used": False, "hook_calls": 2,
                "reads": {}}},
        ]
        server._runner = create_job.CreateRunner(
            jobs_dir=Path(self.tmp.name) / "jobs",
            store=_validated_store(self.tmp.name),
            engine=self.eng)
        self.addCleanup(setattr, server, "_runner", None)
        self.patches = [
            mock.patch.object(server, "NO_MODEL", False),
            mock.patch.object(server, "get_engine", return_value=self.eng),
            mock.patch.object(steer, "calibration_cache_key", return_value={}),
            mock.patch.object(steer, "calibrate_layer_medians",
                              return_value=({5: 2.5}, {"cache_hit": True,
                                                       "key": {"prompts_sha256": "abc"}})),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(server.api)
        self.spec = app.active_spec()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        server._cancel.clear()

    def _events(self, body):
        r = self.client.post("/api/update", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        return [json.loads(line) for line in r.text.splitlines() if line]

    def _explicit(self, **kw):
        body = {"prompt": "Say hi.",
                "layer": 5,
                "direction": [1.0] * self.spec.hidden,
                "scale": 2.0,
                "doses": [0.0, 1.0]}
        body.update(kw)
        return body

    def test_explicit_streams_columns(self):
        ev = self._events(self._explicit())
        self.assertEqual([e["stage"] for e in ev],
                         ["weights", "guard", "columns", "token",
                          "token", "complete"])
        toks = [e for e in ev if e["stage"] == "token"]
        self.assertEqual([(t["col"], t["text"]) for t in toks],
                         [(0, "Hello"), (1, "Hi")])
        res = ev[-1]["result"]
        self.assertEqual([c["text"] for c in res["columns"]],
                         ["Hello", "Hi"])
        self.assertFalse(res["fallback_used"])
        self.assertFalse(res["guard"]["blocked"])
        self.assertFalse(self.eng.lock.locked())

    def test_guard_blocks_update(self):
        self.eng.guard_block = True
        ev = self._events(self._explicit(guard=True))
        self.assertEqual([e["stage"] for e in ev],
                         ["weights", "guard", "blocked"])

    def test_default_update_records_without_blocking(self):
        self.eng.guard_block = True
        ev = self._events(self._explicit())
        self.assertEqual(ev[-1]["stage"], "complete")
        g = ev[-1]["result"]["guard"]
        self.assertFalse(g["blocked"])
        self.assertTrue(g["would_block"])
        self.assertFalse(g["enforced"])

    def test_trait_path_uses_saved_knob(self):
        from safetensors.torch import save_file
        job_id = "blunt-0-20261029-000000-000"
        trait_dir = (Path(self.tmp.name) / "jobs" / job_id / "trait")
        trait_dir.mkdir(parents=True)
        (Path(self.tmp.name) / "jobs" / job_id / "job.json").write_text(
            json.dumps({"id": job_id, "trait": "blunt",
                        "status": "done", "updated": 1.0,
                        "frozen": {"layer": 5, "dose": 1.0}}))
        save_file({"direction": torch.ones(self.spec.hidden)},
                  str(trait_dir / "direction.safetensors"))
        ev = self._events({"prompt": "Say hi.", "trait": "blunt",
                           "doses": [1.0]})
        cols = [e for e in ev if e["stage"] == "columns"][0]
        self.assertEqual(cols["scale"], 2.5)

    def test_input_errors(self):
        hid = self.spec.hidden
        cases = [
            ({"prompt": "x", "trait": "blunt", "layer": 5,
              "direction": [1.0] * hid, "scale": 1.0,
              "doses": [1.0]}, 400),
            ({"prompt": "x", "doses": [1.0]}, 400),
            ({"prompt": "x", "trait": "ghost", "doses": [1.0]}, 404),
            (self._explicit(layer=9999), 400),
            (self._explicit(direction=[1.0, 2.0]), 400),
            (self._explicit(doses=[0.1] * 9), 400),
            (self._explicit(doses=[]), 400),
        ]
        for body, code in cases:
            with self.subTest(body={k: v for k, v in body.items()
                                    if k != "direction"}):
                self.assertEqual(
                    self.client.post("/api/update", json=body).status_code,
                    code)


class _Wave2Base(unittest.TestCase):
    def setUp(self):
        self.eng = FakeEngine([_gen(BASE_WORDS), _gen(STEER_WORDS)])
        self.patches = [
            mock.patch.object(server, "NO_MODEL", False),
            mock.patch.object(server, "reader", None),
            mock.patch.object(server, "get_engine", return_value=self.eng),
            mock.patch.object(steer, "calibration_cache_key", return_value={}),
            mock.patch.object(steer, "calibrate_layer_medians",
                              return_value=({}, {"cache_hit": True,
                                                 "key": {"prompts_sha256": "abc123"}})),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(server.api)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        server._cancel.clear()


class DialStreamTest(_Wave2Base):
    def _script(self):
        done = _gen(STEER_WORDS)
        self.eng.stream_script_single = [
            {"type": "token", "text": "Hi",
             "dial": {"blunt": 0.7}},
            {"type": "token", "text": " there"},
            {"type": "done", "result": done}]

    def test_token_events_carry_dial(self):
        self._script()
        spec = {"trait": "blunt", "layer": 5, "unit": [1.0],
                "b0": 0.0, "b1": 2.0, "job_id": "j"}
        with mock.patch.object(server, "_dial_spec", return_value=spec):
            r = self.client.post("/api/run", json={
                "prompt": "Who are you?", "rows": [ROW],
                "dial_trait": "blunt"})
        self.assertEqual(r.status_code, 200, r.text)
        ev = [json.loads(line) for line in r.text.splitlines() if line]
        self.assertEqual([e["stage"] for e in ev],
                         ["weights", "guard", "baseline", "steered",
                          "token", "token", "complete"])
        self.assertEqual(ev[4]["dial"], {"blunt": 0.7})
        self.assertNotIn("dial", ev[5])
        res = ev[-1]["result"]
        self.assertEqual(res["dial"], {"trait": "blunt", "layer": 5,
                                      "neutral": 0.0, "trait_mean": 2.0})
        run_kw = self.eng.runs[-1][2]
        self.assertEqual(run_kw["dial"], spec)

    def test_unknown_dial_trait_is_error_event(self):
        with mock.patch.object(server, "_dial_spec",
                               side_effect=KeyError("ghost")):
            r = self.client.post("/api/run", json={
                "prompt": "Who are you?", "rows": [ROW],
                "dial_trait": "ghost"})
        ev = [json.loads(line) for line in r.text.splitlines() if line]
        self.assertEqual(ev[-1]["stage"], "error")
        self.assertIn("ghost", ev[-1]["message"])

    def test_no_dial_keeps_old_contract(self):
        ev = [json.loads(line) for line in self.client.post(
            "/api/run", json={"prompt": "Who are you?", "rows": [ROW]}
        ).text.splitlines() if line]
        self.assertEqual([e["stage"] for e in ev],
                         ["weights", "guard", "baseline", "steered",
                          "complete"])
        self.assertIsNone(ev[-1]["result"]["dial"])


class NeuronsApiTest(_Wave2Base):
    def _script(self):
        import app as engine_mod
        n_layers = engine_mod.active_spec().layers
        script = {}
        for L in range(n_layers):
            script[L] = [torch.zeros(5, 8)]
        script[3][0][1, 5] = 9.0
        script[7][0][1, 1] = -7.0
        self.eng.capture_script = script

    def test_top_neurons_and_contexts(self):
        self._script()
        r = self.client.post("/api/neurons", json={
            "prompt": "Say hi.", "rows": [ROW], "token_index": 2,
            "top_k": 2})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertFalse(d["blocked"])
        self.assertEqual(len(d["tokens"]), 5)
        self.assertEqual(
            [(n["layer"], n["neuron"]) for n in d["neurons"]],
            [(3, 5), (7, 1)])
        self.assertAlmostEqual(d["neurons"][0]["activation"], 9.0)
        ctx = d["neurons"][0]["top_contexts"][0]
        self.assertEqual((ctx["position"], ctx["activation"]), (2, 9.0))
        self.assertTrue(ctx["text"])
        texts, layers, point, reduce = self.eng.capture_calls[0]
        self.assertEqual(texts, ["Say hi."])
        self.assertEqual((point, reduce), ("mlp_in", "none"))

    def test_capture_runs_with_edits(self):
        self._script()
        self.client.post("/api/neurons", json={
            "prompt": "Say hi.", "rows": [ROW], "token_index": 1})
        kinds = [c[2] for c in self.eng.capture_calls]
        self.assertIn("mlp_in", kinds)
        cap_runs = [run for run in self.eng.runs
                    if run[0] == "capture"]
        self.assertEqual(len(cap_runs), 1)
        self.assertEqual(len(cap_runs[0][1].neuron), 1)

    def test_bad_index_and_busy(self):
        self._script()
        for bad in (0, 9):
            r = self.client.post("/api/neurons", json={
                "prompt": "Say hi.", "token_index": bad})
            self.assertEqual(r.status_code, 400, bad)
        self.eng.lock.acquire()
        try:
            r = self.client.post("/api/neurons", json={
                "prompt": "Say hi.", "token_index": 0})
        finally:
            self.eng.lock.release()
        self.assertEqual(r.status_code, 409)

    def test_unknown_trait_is_404(self):
        with mock.patch.object(server, "_latest_knob", return_value=None):
            r = self.client.post("/api/neurons", json={
                "prompt": "Say hi.", "trait": "ghost", "token_index": 0})
        self.assertEqual(r.status_code, 404)


class ChatApiTest(_Wave2Base):
    def setUp(self):
        super().setUp()
        server._chats.clear()
        self.addCleanup(server._chats.clear)
        self.eng.stream_script_single = [
            {"type": "token", "text": "Hi"},
            {"type": "done", "result": {"text": "Hi", "token_ids": [1],
                                       "new_tokens": 1,
                                       "cancelled": False}}]

    def _events(self, body):
        r = self.client.post("/api/chat", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        return [json.loads(line) for line in r.text.splitlines() if line]

    def test_two_turns_keep_history_and_edits(self):
        r = self.client.post("/api/chat/start", json={"rows": [ROW]})
        sid = r.json()["session_id"]
        ev1 = self._events({"session_id": sid, "message": "hello"})
        self.assertEqual([e["stage"] for e in ev1], ["token", "done"])
        self.assertEqual(ev1[-1]["result"]["turn"], 1)
        ev2 = self._events({"session_id": sid, "message": "again"})
        self.assertEqual(ev2[-1]["result"]["turn"], 2)
        wraps = self.eng.tokenizer.wrapped_chats
        self.assertEqual(len(wraps[0]), 1)
        self.assertEqual(len(wraps[1]), 3)
        self.assertEqual(wraps[1][1], {"role": "assistant",
                                      "content": "Hi"})
        turns = [run for run in self.eng.runs if "wrapped" in run[2]]
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0][1], turns[1][1])
        hist = self.client.get(f"/api/chat/{sid}").json()["turns"]
        self.assertEqual([(m["role"], m["content"]) for m in hist],
                         [("user", "hello"), ("assistant", "Hi"),
                          ("user", "again"), ("assistant", "Hi")])

    def test_rows_replace_mid_chat(self):
        r = self.client.post("/api/chat/start", json={"rows": [ROW]})
        sid = r.json()["session_id"]
        self._events({"session_id": sid, "message": "hi"})
        other = {"layer": 13, "neuron": 7956, "dose": -1.0}
        self._events({"session_id": sid, "message": "hi", "rows": [other]})
        turns = [run for run in self.eng.runs if "wrapped" in run[2]]
        first = turns[0][1].neuron[0]
        second = turns[1][1].neuron[0]
        self.assertEqual((first.layer, first.neuron), (14, 8949))
        self.assertEqual((second.layer, second.neuron), (13, 7956))

    def test_unknown_chat_and_model_pin(self):
        r = self.client.post("/api/chat", json={"session_id": "nope",
                                                "message": "hi"})
        self.assertEqual(r.status_code, 404)
        r = self.client.post("/api/chat/start", json={})
        sid = r.json()["session_id"]
        import app as engine_mod
        with mock.patch.object(engine_mod, "_active_model_id", "other"):
            r = self.client.post("/api/chat", json={"session_id": sid,
                                                   "message": "hi"})
        self.assertEqual(r.status_code, 409)


class _ReplyTok(FakeTokenizer):
    """Ids follow the word count, so prompt and prompt+reply differ."""

    def __call__(self, text, **kw):
        n = len(text.split())
        return {"input_ids": [list(range(n))]}


class ReaderApiTest(_Wave2Base):
    """R2 routes: neuron card, find, lift, neurons reply (fake engine)."""

    def _stub(self):
        stub = mock.Mock()
        stub.pushes_toward.return_value = {
            "up": [{"token": t, "id": i, "logit": 3.0 - i}
                   for i, t in enumerate(["alpha", "beta", "gamma"])],
            "down": [], "coherence": 0.67, "layer_frac": 0.41}
        stub.fires_on.return_value = [
            {"activation": 4.1, "token": "pirate", "context": "a pirate map",
             "mark": [2, 8]}]
        stub.find_neurons.return_value = {
            "word": "pirate", "token_ids": [7],
            "results": [{"layer": 14, "neuron": 1, "score": 9.0,
                         "rank": 1}]}
        stub.steer_lift.return_value = {
            "words": ["pirate"], "base_p": [0.1], "steered_p": [0.3],
            "lift": 3.0}
        return stub

    def _tmp_root(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def _script(self, layers, rows=5, cols=8):
        script = {}
        for L in range(layers):
            script[L] = [torch.zeros(rows, cols)]
        return script

    def test_neuron_card_shape(self):
        import app as engine_mod
        stub = self._stub()
        root = self._tmp_root()
        slug = server._model_slug(engine_mod._active_model_id)
        cache = root / "store" / "reader" / slug
        cache.mkdir(parents=True)
        (cache / "fires_on.npz").write_bytes(b"stub")
        (cache / "sample.json").write_text(
            json.dumps({"tokens": ["t"] * 100}))
        with mock.patch.object(server, "ROOT", root), \
                mock.patch.object(server, "reader", stub):
            r = self.client.get("/api/neuron/11/1343?k=5")
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d["model_id"], engine_mod._active_model_id)
        self.assertEqual((d["layer"], d["neuron"]), (11, 1343))
        self.assertEqual(d["pushes_toward"]["coherence"], 0.67)
        self.assertEqual(d["fires_on"][0]["token"], "pirate")
        self.assertEqual(d["fires_on_cache"], {"built": True,
                                               "tokens": 100})
        stub.pushes_toward.assert_called_once_with(
            self.eng, 11, 1343, k=5)

    def test_neuron_out_of_bounds_400(self):
        for layer, neuron in ((999, 0), (0, 10**9), (-1, 0)):
            r = self.client.get(f"/api/neuron/{layer}/{neuron}")
            self.assertEqual(r.status_code, 400, (layer, neuron))

    def test_neuron_card_without_cache_has_null_fires_on(self):
        stub = self._stub()
        root = self._tmp_root()
        with mock.patch.object(server, "ROOT", root), \
                mock.patch.object(server, "reader", stub):
            r = self.client.get("/api/neuron/11/1343")
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertIsNone(d["fires_on"])
        self.assertEqual(d["fires_on_cache"], {"built": False,
                                               "tokens": 0})
        stub.fires_on.assert_not_called()

    def test_find_parses_layer_range(self):
        stub = self._stub()
        with mock.patch.object(server, "reader", stub):
            r = self.client.get(
                "/api/find?word=pirate&layers=8-27&k=5")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["word"], "pirate")
        _, args, kw = stub.find_neurons.mock_calls[0]
        self.assertEqual(args[1], "pirate")
        self.assertEqual(args[2], list(range(8, 28)))
        self.assertEqual(kw["k"], 5)

    def test_find_bad_range_400(self):
        for layers in ("27-8", "abc", "0-999", "8-"):
            r = self.client.get(
                f"/api/find?word=pirate&layers={layers}")
            self.assertEqual(r.status_code, 400, layers)

    def test_find_empty_word_400(self):
        r = self.client.get("/api/find?word=%20%20")
        self.assertEqual(r.status_code, 400)
        stub = self._stub()
        stub.find_neurons.side_effect = ValueError("no tokens")
        with mock.patch.object(server, "reader", stub):
            r = self.client.get("/api/find?word=zzz")
        self.assertEqual(r.status_code, 400)

    def test_neurons_reply_tokens_addressable(self):
        import app as engine_mod
        n_layers = engine_mod.active_spec().layers
        self.eng.tokenizer = _ReplyTok()
        script = {L: torch.zeros(2, 8) for L in range(n_layers)}
        script[2][1, 3] = 9.0
        with mock.patch.object(server, "_capture_reply_mats",
                               return_value=script) as cap:
            r = self.client.post("/api/neurons", json={
                "prompt": "Say hi.", "rows": [ROW], "token_index": 2,
                "top_k": 1, "reply": "hello there"})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(len(d["tokens"]), 3)
        self.assertEqual(d["reply_start"], 2)
        self.assertEqual(
            [(n["layer"], n["neuron"]) for n in d["neurons"]],
            [(2, 3)])
        self.assertEqual(self.eng.capture_calls, [])
        cap.assert_called_once()
        self.assertEqual(list(cap.call_args[0][3]), [0, 1, 2])

    def test_neurons_reply_char_maps_to_token(self):
        class Tok:
            pieces = ["<s>", "Q", "Hello", " there", " friend"]
            def decode(self, ids, skip_special_tokens=True):
                return "".join(self.pieces[i] for i in ids)
        ids = [0, 1, 2, 3, 4]
        # reply "Hello there friend" starts at token 2
        self.assertEqual(server._reply_token_at(Tok(), ids, 2, 0), 2)
        self.assertEqual(server._reply_token_at(Tok(), ids, 2, 4), 2)
        self.assertEqual(server._reply_token_at(Tok(), ids, 2, 5), 3)  # the space joins " there"
        self.assertEqual(server._reply_token_at(Tok(), ids, 2, 7), 3)
        self.assertEqual(server._reply_token_at(Tok(), ids, 2, 13), 4)

    def test_neurons_needs_exactly_one_index(self):
        for body in ({}, {"token_index": 1, "reply_char": 0, "reply": "hi"},
                     {"reply_char": 5, "reply": "hi"}):
            r = self.client.post("/api/neurons", json={"prompt": "Say hi.", **body})
            self.assertEqual(r.status_code, 400, body)

    def test_neurons_without_reply_unchanged(self):
        import app as engine_mod
        n_layers = engine_mod.active_spec().layers
        script = self._script(n_layers)
        script[3][0][1, 5] = 9.0
        script[7][0][1, 1] = -7.0
        self.eng.capture_script = script
        r = self.client.post("/api/neurons", json={
            "prompt": "Say hi.", "rows": [ROW], "token_index": 2,
            "top_k": 2})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertIsNone(d["reply_start"])
        self.assertEqual(len(d["tokens"]), 5)
        self.assertEqual(
            [(n["layer"], n["neuron"]) for n in d["neurons"]],
            [(3, 5), (7, 1)])
        self.assertAlmostEqual(d["neurons"][0]["activation"], 9.0)
        self.assertTrue(all(n["pushes_toward_top"] is None
                            for n in d["neurons"]))

    def test_neurons_pushes_toward_top_null_without_reader(self):
        import app as engine_mod
        n_layers = engine_mod.active_spec().layers
        self.eng.capture_script = self._script(n_layers)
        r = self.client.post("/api/neurons", json={
            "prompt": "Say hi.", "token_index": 1, "top_k": 2})
        self.assertEqual(r.status_code, 200, r.text)
        tops = [n["pushes_toward_top"] for n in r.json()["neurons"]]
        self.assertTrue(tops)
        self.assertTrue(all(t is None for t in tops))

    def test_neurons_pushes_toward_top_with_reader(self):
        import app as engine_mod
        n_layers = engine_mod.active_spec().layers
        self.eng.capture_script = self._script(n_layers)
        with mock.patch.object(server, "reader", self._stub()):
            r = self.client.post("/api/neurons", json={
                "prompt": "Say hi.", "token_index": 1, "top_k": 1})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["neurons"][0]["pushes_toward_top"],
                         ["alpha", "beta", "gamma"])

    def test_lift_shape(self):
        stub = self._stub()
        with mock.patch.object(server, "reader", stub):
            r = self.client.post("/api/lift", json={
                "layer": 14, "neuron": 100, "dose": 1.5,
                "words": ["pirate"], "prompts": None})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["lift"], 3.0)
        _, args, _ = stub.steer_lift.mock_calls[0]
        self.assertEqual(tuple(args[1:4]), (14, 100, 1.5))
        self.assertEqual(args[4], ["pirate"])
        self.assertEqual(len(args[5]), 8)

    def test_lift_dose_out_of_range_422(self):
        r = self.client.post("/api/lift", json={
            "layer": 14, "neuron": 100, "dose": 99.0,
            "words": ["pirate"]})
        self.assertEqual(r.status_code, 422)

    def test_model_routes_503_without_weights(self):
        with mock.patch.object(server, "NO_MODEL", True):
            paths = [("/api/neuron/11/1343", "get"),
                     ("/api/find?word=pirate", "get")]
            for path, _ in paths:
                r = self.client.get(path)
                self.assertEqual(r.status_code, 503, path)
            r = self.client.post("/api/lift", json={
                "layer": 14, "neuron": 100, "dose": 1.0,
                "words": ["pirate"]})
            self.assertEqual(r.status_code, 503)
            r = self.client.post("/api/neurons", json={
                "prompt": "Say hi.", "token_index": 1})
            self.assertEqual(r.status_code, 503)

    def test_model_routes_409_when_busy(self):
        self.eng.lock.acquire()
        try:
            for path in ("/api/neuron/11/1343",
                         "/api/find?word=pirate"):
                r = self.client.get(path)
                self.assertEqual(r.status_code, 409, path)
            r = self.client.post("/api/lift", json={
                "layer": 14, "neuron": 100, "dose": 1.0,
                "words": ["pirate"]})
            self.assertEqual(r.status_code, 409)
            r = self.client.post("/api/neurons", json={
                "prompt": "Say hi.", "token_index": 1})
            self.assertEqual(r.status_code, 409)
        finally:
            self.eng.lock.release()

    def test_neuron_card_replay_from_cards_json(self):
        import app as engine_mod
        root = self._tmp_root()
        slug = server._model_slug(engine_mod._active_model_id)
        cards = root / "store" / "reader" / slug
        cards.mkdir(parents=True)
        card = {"model_id": engine_mod._active_model_id, "layer": 11,
                "neuron": 1343, "pushes_toward": {"up": []},
                "fires_on": None,
                "fires_on_cache": {"built": False, "tokens": 0}}
        (cards / "cards.json").write_text(json.dumps({"11:1343": card}))
        with mock.patch.object(server, "REPLAY", True), \
                mock.patch.object(server, "ROOT", root):
            r = self.client.get("/api/neuron/11/1343")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), card)

    def test_neuron_card_replay_without_cards_503(self):
        root = self._tmp_root()
        with mock.patch.object(server, "REPLAY", True), \
                mock.patch.object(server, "ROOT", root):
            r = self.client.get("/api/neuron/11/1343")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(
            r.json()["detail"], "Neuron cards need the model, or a "
                                "recorded cards.json for replay mode.")

    def test_reader_routes_reject_cross_origin_post(self):
        evil = {"Origin": "http://evil.test"}
        r = self.client.post("/api/lift", json={
            "layer": 14, "neuron": 100, "dose": 1.0,
            "words": ["pirate"]}, headers=evil)
        self.assertEqual(r.status_code, 403)
        r = self.client.post("/api/neurons", json={
            "prompt": "Say hi.", "token_index": 1}, headers=evil)
        self.assertEqual(r.status_code, 403)


if __name__ == "__main__":
    unittest.main()
