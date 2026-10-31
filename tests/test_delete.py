"""Unit tests: run-time Delete (DL).

No model. Proves: /api/delete streams off/on columns with the
ablation applied to the on-column; a guard block ends the stream
first; /api/delete/check returns the card + nll + refusal +
guard shape; traits without a knob are 422 with the eligibility
sentence; the traits list marks eligibility; busy and missing
weights behave like the update route.
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
from model_lobotomy import trait_eval as T  # noqa: E402
from model_lobotomy.engine import EMPTY_EDITS  # noqa: E402

KNOB = {"layer": 17, "dose": 1.5, "direction": [0.1, 0.2],
        "job_id": "job-1"}


class FakeDeleteEngine:
    backend = "torch"

    def supports(self, feature):
        return True

    def __init__(self):
        self.lock = threading.Lock()
        self.model = object()
        self.tokenizer = object()
        self.model_id = "Qwen/Qwen3.5-4B"
        self.revision = "stub-rev"
        self.calls = []

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
        ablated = len(getattr(edits, "ablate", ())) > 0
        text = "plain" if not ablated else "scrubbed"
        yield {"type": "token", "text": text}
        yield {"type": "done", "result": {"text": text,
                                         "token_ids": [1],
                                         "new_tokens": 1,
                                         "cancelled": False}}


class DeleteCase(unittest.TestCase):
    def setUp(self):
        self.eng = FakeDeleteEngine()
        self.patches = [
            mock.patch.object(server, "NO_MODEL", False),
            mock.patch.object(server, "get_engine",
                              return_value=self.eng),
            mock.patch.object(server, "_latest_knob",
                              return_value=dict(KNOB)),
            mock.patch.object(server, "_verified_guard",
                              return_value=SimpleNamespace(
                                  blocked=False)),
            mock.patch.object(server, "_dial_spec",
                              side_effect=KeyError("pirate")),
            mock.patch.object(server, "_calibrate",
                              return_value=({17: 2.0},
                                            {"cache_hit": True})),
            mock.patch.object(server, "_guard_event_dict",
                              return_value={"blocked": False}),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(server.api)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        server._cancel.clear()

    def _events(self, body):
        r = self.client.post("/api/delete", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        return [json.loads(line) for line in r.text.splitlines()
                if line]


class DeleteStreamTest(DeleteCase):
    def test_delete_streams_two_columns(self):
        evs = self._events({"prompt": "Say hi.", "trait": "pirate"})
        kinds = [e["stage"] for e in evs]
        self.assertEqual(kinds[0], "weights")
        self.assertEqual(kinds[1], "guard")
        self.assertEqual(evs[2]["stage"], "columns")
        self.assertEqual(evs[2]["modes"], ["off", "on"])
        tokens = [e for e in evs if e["stage"] == "token"]
        self.assertEqual([(t["col"], t["mode"], t["text"])
                          for t in tokens],
                         [(0, "off", "plain"), (1, "on", "scrubbed")])
        done = evs[-1]["result"]
        self.assertEqual(
            [(c["mode"], c["text"]) for c in done["columns"]],
            [("off", "plain"), ("on", "scrubbed")])
        self.assertEqual(done["trait"], "pirate")
        self.assertEqual(done["layer"], 17)

    def test_ablation_actually_applies(self):
        self._events({"prompt": "Say hi.", "trait": "pirate"})
        self.assertEqual(len(self.eng.calls), 2)
        off_edits = self.eng.calls[0][1]
        on_edits = self.eng.calls[1][1]
        self.assertEqual(off_edits, EMPTY_EDITS)
        self.assertEqual(len(on_edits.ablate), 1)
        self.assertEqual(on_edits.ablate[0].layer, 17)
        self.assertEqual(tuple(on_edits.ablate[0].direction),
                         (0.1, 0.2))

    def test_delete_guard_block_first(self):
        with mock.patch.object(
                server, "_verified_guard",
                return_value=SimpleNamespace(blocked=True)):
            evs = self._events({"prompt": "Say hi.",
                                "trait": "pirate"})
        self.assertEqual(
            [e["stage"] for e in evs],
            ["weights", "guard", "blocked"])
        self.assertEqual(self.eng.calls, [])

    def test_ineligible_trait_422(self):
        with mock.patch.object(server, "_latest_knob",
                               return_value=None):
            r = self.client.post("/api/delete", json={
                "prompt": "Say hi.", "trait": "nope"})
        self.assertEqual(r.status_code, 422)
        self.assertIn("not eligible", r.text)

    def test_delete_503_and_409(self):
        with mock.patch.object(server, "NO_MODEL", True):
            r = self.client.post("/api/delete", json={
                "prompt": "Say hi.", "trait": "pirate"})
            self.assertEqual(r.status_code, 503)
        self.eng.lock.acquire()
        try:
            r = self.client.post("/api/delete", json={
                "prompt": "Say hi.", "trait": "pirate"})
            self.assertEqual(r.status_code, 409)
        finally:
            self.eng.lock.release()


class DeleteCheckTest(DeleteCase):
    def test_check_card_shape(self):
        out = {"trait": "pirate", "layer": 17, "n": 2,
               "gains": [0.5, 0.4], "control_gains": [0.0, 0.1],
               "card": {"passed": True}, "nll": {"base_mean": 1.0,
                                                "deleted_mean": 1.2,
                                                "delta": 0.2},
               "refusal": {"base_rate": 1.0, "deleted_rate": 1.0,
                           "n": 16},
               "guard": {"blocked": False}}
        with mock.patch.object(T, "delete_check",
                               return_value=dict(out)) as dc, \
                mock.patch.object(T, "family_cfg",
                                  return_value={"name": "pirate"}):
            r = self.client.post("/api/delete/check", json={
                "trait": "pirate"})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d["trait"], "pirate")
        self.assertIn("seconds", d)
        self.assertIn("delta", d["nll"])
        dc.assert_called_once()
        _, kwargs = dc.call_args
        self.assertIsNone(kwargs["prompts"])

    def test_check_prompts_validated(self):
        with mock.patch.object(T, "delete_check",
                               return_value={}), \
                mock.patch.object(T, "family_cfg",
                                  return_value={"name": "pirate"}):
            r = self.client.post("/api/delete/check", json={
                "trait": "pirate", "prompts": ["  "]})
            self.assertEqual(r.status_code, 400)
            r = self.client.post("/api/delete/check", json={
                "trait": "pirate",
                "prompts": [f"q{i}" for i in range(25)]})
            self.assertEqual(r.status_code, 400)

    def test_check_ineligible_422(self):
        with mock.patch.object(server, "_latest_knob",
                               return_value=None):
            r = self.client.post("/api/delete/check", json={
                "trait": "nope"})
        self.assertEqual(r.status_code, 422)


class DeleteCheckFnTest(unittest.TestCase):
    def test_delete_check_counts_and_shape(self):
        eng = mock.Mock()
        eng.model = object()
        eng.tokenizer = object()
        spec = SimpleNamespace()
        cfg = {"name": "pirate", "pos": "POS", "neg": "NEG",
               "judge": "J"}
        knob = {"layer": 3, "direction": [0.5, -0.5]}
        texts = [f"t{i}" for i in range(60)]
        with mock.patch.object(T, "_gen",
                               side_effect=texts) as gen, \
                mock.patch.object(T, "judge_score",
                                  return_value=0.7), \
                mock.patch.object(T.spike, "nll",
                                  return_value=1.0), \
                mock.patch.object(
                    T, "_shuffled_unit",
                    return_value=SimpleNamespace(
                        tolist=lambda: [0.1, 0.1])), \
                mock.patch(
                    "model_lobotomy.guard.refusal_judge_fn",
                    return_value=lambda text: 1.0), \
                mock.patch(
                    "model_lobotomy.guard.Guard") as guard_cls, \
                mock.patch(
                    "model_lobotomy.guard.verdict_dict",
                    return_value={"blocked": False}):
            guard_cls.return_value.check_verified.return_value = (
                SimpleNamespace(blocked=False))
            out = T.delete_check(
                eng, spec, cfg, knob, {3: 1.0},
                [f"q{i}" for i in range(120)],
                prompts=["a", "b"])
        self.assertEqual(out["n"], 2)
        self.assertEqual(len(out["gains"]), 2)
        self.assertEqual(len(out["control_gains"]), 2)
        self.assertEqual(out["nll"]["delta"], 0.0)
        self.assertEqual(out["refusal"]["deleted_rate"], 1.0)
        self.assertEqual(out["refusal"]["n"], 8)
        # 2 prompts x (3 trait + 2 nll) + 8 refusal x 2.
        self.assertEqual(gen.call_count, 2 * 5 + 8 * 2)

    def test_delete_check_defaults_to_bank(self):
        eng = mock.Mock()
        eng.model = object()
        eng.tokenizer = object()
        spec = SimpleNamespace()
        cfg = {"name": "pirate", "pos": "POS", "neg": "NEG",
               "judge": "J"}
        knob = {"layer": 3, "direction": [0.5, -0.5]}
        texts = [f"t{i}" for i in range(200)]
        with mock.patch.object(T, "_gen",
                               side_effect=texts), \
                mock.patch.object(T, "judge_score",
                                  return_value=0.7), \
                mock.patch.object(T.spike, "nll",
                                  return_value=1.0), \
                mock.patch.object(
                    T, "_shuffled_unit",
                    return_value=SimpleNamespace(
                        tolist=lambda: [0.1, 0.1])), \
                mock.patch(
                    "model_lobotomy.guard.refusal_judge_fn",
                    return_value=lambda text: 1.0), \
                mock.patch(
                    "model_lobotomy.guard.Guard"), \
                mock.patch(
                    "model_lobotomy.guard.verdict_dict",
                    return_value={"blocked": False}):
            out = T.delete_check(
                eng, spec, cfg, knob, {3: 1.0},
                ["", "  "] + [f"q{i}" for i in range(30)])
        self.assertEqual(out["n"], T.DELETE_CHECK_PROMPTS)
        with self.assertRaises(ValueError):
            with mock.patch.object(
                    T, "_shuffled_unit",
                    return_value=SimpleNamespace(
                        tolist=lambda: [0.1, 0.1])):
                T.delete_check(eng, spec, cfg, knob, {3: 1.0},
                               ["", "  "])


if __name__ == "__main__":
    unittest.main()
