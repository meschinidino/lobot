"""Unit tests: FastAPI server stream contract (server.py).

Value: protects=NDJSON stage order, split char offsets, lock release,
Stop, input errors; fails_when=the frontend contract drifts or the run lock
leaks after an error. no_claim=real generation (the engine is stubbed
here; the model-backed path stays in the opt-in integration lane).
"""
from __future__ import annotations

import json
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app  # noqa: E402
import server  # noqa: E402
import steer  # noqa: E402

BASE_WORDS = "I am Qwen , a model".split()
STEER_WORDS = "I am an AI that wants".split()
VOCAB = BASE_WORDS + STEER_WORDS


class FakeTokenizer:
    """Token id i decodes to VOCAB[i]; words join with single spaces."""

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(VOCAB[i] for i in ids)


def _gen(words):
    ids = [VOCAB.index(w) if w in BASE_WORDS else len(BASE_WORDS) + STEER_WORDS.index(w)
           for w in words]
    return {"text": " ".join(words), "new_tokens": len(ids), "token_ids": ids,
            "seconds": 0.1, "prefill_s": 0.01, "decode_s": 0.09, "tok_per_s": 50.0,
            "peak_mib": 1.0, "driver_mib": 1.0}


ROW = {"layer": 14, "neuron": 8949, "dose": 1.0}


class ServerStreamTest(unittest.TestCase):
    def setUp(self):
        outputs = iter([_gen(BASE_WORDS), _gen(STEER_WORDS)])
        self.patches = [
            mock.patch.object(server, "NO_MODEL", False),
            mock.patch.object(app, "load_engine", return_value=(object(), FakeTokenizer(), 1)),
            mock.patch.object(steer, "calibration_cache_key", return_value={}),
            mock.patch.object(steer, "calibrate_layer_medians",
                              return_value=({}, {"cache_hit": True,
                                                 "key": {"prompts_sha256": "abc123"}})),
            mock.patch.object(steer, "generate_text", side_effect=lambda *a, **k: next(outputs)),
            mock.patch.object(steer, "register_steering_hooks"),
            mock.patch.object(steer, "remove_steering_hooks"),
            mock.patch.object(steer, "hook_call_count", return_value=3),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(server.api)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def _events(self, body):
        r = self.client.post("/api/run", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        return [json.loads(line) for line in r.text.splitlines() if line]

    def test_stage_order_and_split_offsets(self):
        ev = self._events({"prompt": "Who are you?", "rows": [ROW]})
        self.assertEqual([e["stage"] for e in ev], ["weights", "baseline", "steered", "complete"])
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
        self.assertFalse(app._run_lock.locked())

    def test_hooks_not_fired_is_error_and_releases_lock(self):
        with mock.patch.object(steer, "hook_call_count", return_value=0):
            ev = self._events({"prompt": "Who are you?", "rows": [ROW]})
        self.assertEqual(ev[-1]["stage"], "error")
        self.assertIn("did not fire", ev[-1]["message"])
        self.assertFalse(app._run_lock.locked())

    def test_engine_exception_is_error_and_releases_lock(self):
        with mock.patch.object(app, "load_engine", side_effect=RuntimeError("disk full")):
            ev = self._events({"prompt": "Who are you?", "rows": [ROW]})
        self.assertEqual(ev[-1], {"stage": "error", "message": "Run failed: disk full"})
        self.assertFalse(app._run_lock.locked())

    def test_busy_lock_returns_409(self):
        app._run_lock.acquire()
        try:
            r = self.client.post("/api/run", json={"prompt": "x", "rows": [ROW]})
        finally:
            app._run_lock.release()
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
        def gen(*a, **k):
            self.assertIs(k["cancel"], server._cancel)
            server._cancel.set()  # the user presses Stop mid-baseline
            return _gen(BASE_WORDS)
        with mock.patch.object(steer, "generate_text", side_effect=gen) as g:
            ev = self._events({"prompt": "Who are you?", "rows": [ROW]})
        self.assertEqual(ev[-1], {"stage": "stopped", "after": "baseline"})
        self.assertNotIn("steered", [e["stage"] for e in ev])
        self.assertEqual(g.call_count, 1)
        self.assertFalse(app._run_lock.locked())

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
        self.assertFalse(app._run_lock.locked())

    def test_stop_when_idle_reports_not_stopping(self):
        self.assertEqual(self.client.post("/api/stop").json(), {"stopping": False})
        server._cancel.clear()

    def test_no_model_mode_returns_503(self):
        with mock.patch.object(server, "NO_MODEL", True):
            r = self.client.post("/api/run", json={"prompt": "x", "rows": [ROW]})
        self.assertEqual(r.status_code, 503)


class ReplayModeTest(unittest.TestCase):
    """Replay mode answers recorded setups only, with the run contract and no weights."""

    def setUp(self):
        self.ex = json.loads((ROOT / "web" / "example.json").read_text())
        self.patches = [mock.patch.object(server, "REPLAY", True),
                        mock.patch.object(server, "REPLAY_STAGE_S", 0.0),
                        mock.patch.object(app, "load_engine", side_effect=AssertionError("weights loaded")),
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
        self.assertFalse(app._run_lock.locked())

    def test_unrecorded_setup_is_refused(self):
        rows = [{**self.ex["rows"][0], "dose": 0.5}]
        r = self.client.post("/api/run", json={"prompt": self.ex["prompt"], "rows": rows})
        self.assertEqual(r.status_code, 404)
        self.assertIn("not recorded", r.json()["detail"])

    def test_stop_during_replay_stops_and_releases_lock(self):
        server._cancel.set()
        gen = server._replay_stream(server._recordings()[0])
        events = [json.loads(b) for b in gen]
        self.assertEqual(events[-1], {"stage": "stopped", "after": "weights"})
        self.assertFalse(app._run_lock.locked())

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


if __name__ == "__main__":
    unittest.main()
