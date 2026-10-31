"""Unit tests: chat with live dials (CH).

No model. Proves: chat token events carry every dial trait;
a lagging tap omits only that trait; done.result and history
carry per-turn dial lists; a transcript replays exactly against
a deterministic scripted engine and flags a diverged turn; the
single-message chat wrap equals the run wrap; unknown dial
traits 404 and too many 400; replay pins the model.
"""
from __future__ import annotations

import json
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import server  # noqa: E402
from model_lobotomy import steer  # noqa: E402

ROW = {"layer": 14, "neuron": 8949, "dose": 1.0}


class FakeTok:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        self.calls.append((messages, tokenize,
                           add_generation_prompt, kw))
        return "wrapped:" + "|".join(m["content"] for m in messages)


class FakeChatEngine:
    """Scripted run_stream: script_for(call_index, prompt) -> events."""

    backend = "torch"

    def supports(self, feature):
        return True

    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.model = object()
        self.model_id = "Qwen/Qwen3.5-4B"
        self.revision = "stub-rev"
        self.tokenizer = FakeTok()
        self.calls = []
        self.script_for = lambda i, prompt: [
            {"type": "token", "text": "Hi"},
            {"type": "done", "result": {"text": "Hi", "token_ids": [1],
                                       "new_tokens": 1,
                                       "cancelled": False}}]

    def load(self, model_id):
        self.model_id = model_id
        return 1

    @staticmethod
    def clean():
        from contextlib import contextmanager

        @contextmanager
        def _clean():
            yield

        return _clean()

    def run_stream(self, prompt, edits, **kw):
        self.calls.append((prompt, edits, kw))
        for ev in self.script_for(len(self.calls) - 1, prompt):
            yield dict(ev)


def _knob(layer=17):
    return {"layer": layer, "dose": 1.5, "direction": [0.1, 0.2],
            "job_id": "job-1"}


def _spec(trait, layer=17):
    return {"trait": trait, "layer": layer, "unit": [0.6, 0.8],
            "b0": 0.0, "b1": 1.0, "job_id": "job-1"}


class ChatDialsCase(unittest.TestCase):
    def setUp(self):
        self.eng = FakeChatEngine()
        self.patches = [
            mock.patch.object(server, "NO_MODEL", False),
            mock.patch.object(server, "get_engine",
                              return_value=self.eng),
            mock.patch.object(server, "_latest_knob",
                              return_value=_knob()),
            mock.patch.object(server, "_dial_spec",
                              side_effect=lambda eng, t: _spec(t)),
            mock.patch.object(server, "_verified_guard",
                              return_value=SimpleNamespace(
                                  blocked=False)),
            mock.patch.object(steer, "calibration_cache_key",
                              return_value={}),
            mock.patch.object(
                steer, "calibrate_layer_medians",
                return_value=({14: 1.0, 17: 2.0},
                              {"cache_hit": True,
                               "key": {"prompts_sha256": "abc"}})),
        ]
        for p in self.patches:
            p.start()
        server._chats.clear()
        self.addCleanup(server._chats.clear)
        self.client = TestClient(server.api)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        server._cancel.clear()

    def _events(self, body):
        r = self.client.post("/api/chat", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        return [json.loads(line) for line in r.text.splitlines()
                if line]

    def _start(self, **kw):
        r = self.client.post("/api/chat/start", json=kw)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["session_id"]


class DialEventsTest(ChatDialsCase):
    def test_token_events_carry_many_dials(self):
        sid = self._start(rows=[ROW],
                          dial_traits=["pirate", "cheerful"])
        self.eng.script_for = lambda i, prompt: [
            {"type": "token", "text": "Yo",
             "dial": {"pirate": 0.8, "cheerful": 0.2}},
            {"type": "done", "result": {"text": "Yo", "token_ids": [1],
                                       "new_tokens": 1,
                                       "cancelled": False}}]
        evs = self._events({"session_id": sid, "message": "hi"})
        self.assertEqual(evs[0]["stage"], "token")
        self.assertEqual(evs[0]["dial"],
                         {"pirate": 0.8, "cheerful": 0.2})
        done = evs[-1]["result"]
        self.assertEqual(done["dials"]["pirate"],
                         [{"text": "Yo", "value": 0.8}])
        self.assertEqual(done["dials"]["cheerful"],
                         [{"text": "Yo", "value": 0.2}])
        # run_stream got both specs; edits carry both Read taps.
        _prompt, edits, kw = self.eng.calls[0]
        self.assertEqual(len(kw["dials"]), 2)
        self.assertEqual(len(edits.read), 2)

    def test_lagging_tap_omits_that_trait_only(self):
        sid = self._start(dial_traits=["pirate", "cheerful"])
        self.eng.script_for = lambda i, prompt: [
            {"type": "token", "text": "Yo",
             "dial": {"pirate": 0.8}},
            {"type": "done", "result": {"text": "Yo", "token_ids": [1],
                                       "new_tokens": 1,
                                       "cancelled": False}}]
        evs = self._events({"session_id": sid, "message": "hi"})
        self.assertEqual(evs[0]["dial"], {"pirate": 0.8})
        done = evs[-1]["result"]
        self.assertEqual(done["dials"]["cheerful"],
                         [{"text": "Yo", "value": None}])

    def test_no_dials_no_dial_keys(self):
        sid = self._start(rows=[ROW])
        evs = self._events({"session_id": sid, "message": "hi"})
        self.assertNotIn("dial", evs[0])
        self.assertEqual(evs[-1]["result"]["dials"], {})

    def test_history_returns_dial_turns(self):
        sid = self._start(dial_traits=["pirate"])
        self.eng.script_for = lambda i, prompt: [
            {"type": "token", "text": f"t{i}",
             "dial": {"pirate": 0.1 * (i + 1)}},
            {"type": "done", "result": {"text": f"t{i}",
                                       "token_ids": [1],
                                       "new_tokens": 1,
                                       "cancelled": False}}]
        self._events({"session_id": sid, "message": "one"})
        self._events({"session_id": sid, "message": "two"})
        hist = self.client.get(f"/api/chat/{sid}").json()
        self.assertEqual(hist["dial_traits"], ["pirate"])
        self.assertEqual(len(hist["dial_turns"]), 2)
        self.assertEqual(hist["dial_turns"][1]["dials"]["pirate"],
                         [{"text": "t1", "value": 0.2}])

    def test_unknown_dial_trait_404(self):
        with mock.patch.object(server, "_latest_knob",
                               return_value=None):
            r = self.client.post("/api/chat/start", json={
                "dial_traits": ["nope"]})
        self.assertEqual(r.status_code, 404)

    def test_too_many_dial_traits_400(self):
        r = self.client.post("/api/chat/start", json={
            "dial_traits": ["a", "b", "c", "d", "e"]})
        self.assertEqual(r.status_code, 400)


class WrapTest(unittest.TestCase):
    def test_single_wrap_equals_run_wrap(self):
        tok = FakeTok()
        spec = SimpleNamespace(thinking_kw=False)
        chat = server._wrap_chat(
            tok, [{"role": "user", "content": "hi"}], spec)
        run = steer.wrap_prompt_notk(tok, "hi", spec)
        self.assertEqual(chat, run)
        self.assertEqual(tok.calls[0], tok.calls[1])
        think = SimpleNamespace(thinking_kw=True)
        chat_t = server._wrap_chat(
            tok, [{"role": "user", "content": "hi"}], think)
        run_t = steer.wrap_prompt_notk(tok, "hi", think)
        self.assertEqual(chat_t, run_t)


class ReplayTest(ChatDialsCase):
    def _deterministic_script(self):
        def script(i, prompt):
            return [
                {"type": "token", "text": f"r{i}",
                 "dial": {"pirate": round(0.1 * (i + 1), 2)}},
                {"type": "done", "result": {"text": f"r{i}",
                                           "token_ids": [1],
                                           "new_tokens": 1,
                                           "cancelled": False}}]

        self.eng.script_for = script

    def test_transcript_round_trip_matches(self):
        from model_lobotomy import app as engine_mod
        sid = self._start(rows=[ROW], dial_traits=["pirate"])
        self._deterministic_script()
        self._events({"session_id": sid, "message": "one"})
        other = {"layer": 13, "neuron": 7956, "dose": -1.0}
        self._events({"session_id": sid, "message": "two",
                      "rows": [other]})
        transcript = self.client.get(
            f"/api/chat/{sid}/transcript").json()
        self.assertEqual(transcript["model_id"],
                         engine_mod._active_model_id)
        self.assertEqual(len(transcript["turns"]), 2)
        self.assertEqual(transcript["turns"][1]["rows"], [other])
        # Fresh script state: replay re-runs both turns.
        self.eng.calls.clear()
        r = self.client.post("/api/chat/replay",
                             json={"transcript": transcript})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertTrue(d["match"], d)
        self.assertEqual([t["match"] for t in d["turns"]],
                         [True, True])
        self.assertTrue(all(t["dials_max_diff"] == 0.0
                            for t in d["turns"]))
        self.assertNotEqual(d["session_id"], sid)

    def test_replay_flags_divergence(self):
        sid = self._start(dial_traits=["pirate"])
        self._deterministic_script()
        self._events({"session_id": sid, "message": "one"})
        self._events({"session_id": sid, "message": "two"})
        transcript = self.client.get(
            f"/api/chat/{sid}/transcript").json()

        def diverged(i, prompt):
            text = f"r{i}" if i == 0 else "CHANGED"
            return [
                {"type": "token", "text": text,
                 "dial": {"pirate": 0.5}},
                {"type": "done", "result": {"text": text,
                                           "token_ids": [1],
                                           "new_tokens": 1,
                                           "cancelled": False}}]

        self.eng.script_for = diverged
        self.eng.calls.clear()
        r = self.client.post("/api/chat/replay",
                             json={"transcript": transcript})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertFalse(d["match"])
        self.assertEqual(
            [(t["turn"], t["match"], t["reply_match"])
             for t in d["turns"]],
            [(1, False, True), (2, False, False)])

    def test_replay_model_pin_409(self):
        r = self.client.post("/api/chat/replay", json={
            "transcript": {"model_id": "other",
                           "turns": [{"message": "hi"}]}})
        self.assertEqual(r.status_code, 409)

    def test_replay_empty_transcript_400(self):
        r = self.client.post("/api/chat/replay",
                             json={"transcript": {"turns": []}})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
