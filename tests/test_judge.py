"""Unit tests: judge interface (trait_eval.py) and validate_judge.py.

No network. The Meta API is mocked at urllib.request.urlopen.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import trait_eval as T  # noqa: E402

import torch  # noqa: E402


def _load_validate_script():
    spec = importlib.util.spec_from_file_location(
        "validate_judge_mod", ROOT / "scripts" / "validate_judge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Tok:
    def encode(self, w, add_special_tokens=False):
        return [5] if w == "Yes" else [7]

    def apply_chat_template(self, msgs, **kw):
        return "WRAPPED"

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2]])}


class _Model:
    def __init__(self, yes_logit=2.0):
        self.yes_logit = yes_logit

    def __call__(self, ids):
        from types import SimpleNamespace
        logits = torch.zeros(1, 3, 10)
        logits[0, -1, 5] = self.yes_logit
        return SimpleNamespace(logits=logits)


class LocalJudgeTest(unittest.TestCase):
    def test_local_judge_unchanged(self):
        for yes_logit, reply in ((2.0, "r."), (0.0, "r."),
                                 (-1.0, "another reply")):
            with self.subTest(yes_logit=yes_logit, reply=reply):
                with mock.patch.object(T.steer, "DEVICE", "cpu"):
                    want = T.judge_score(_Model(yes_logit), _Tok(),
                                         None, "q?", reply, "blunt")
                    got = T.LocalJudge(_Model(yes_logit), _Tok(),
                                       None).score("q?", reply, "blunt")
                self.assertAlmostEqual(got, want, places=6)

    def test_local_judge_refusal_is_zero(self):
        with mock.patch.object(T.steer, "DEVICE", "cpu"):
            got = T.LocalJudge(_Model(), _Tok(), None).score(
                "q?", "Sorry, I can't help.", "blunt")
        self.assertEqual(got, 0.0)


def _payload(text, status="completed"):
    return {"status": status,
            "output": [
                {"type": "reasoning", "status": "completed"},
                {"type": "message", "role": "assistant",
                 "status": "completed",
                 "content": [{"type": "output_text", "text": text,
                              "annotations": [], "logprobs": []}]}],
            "usage": {"input_tokens": 43, "output_tokens": 190,
                      "total_tokens": 233}}


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code, body=b'{"error": {"message": "x"}}'):
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(body))


class MetaJudgeParseTest(unittest.TestCase):
    def _judge(self, **kw):
        args = {"key": "fake-key", "model": "fake-model"}
        args.update(kw)
        return T.MetaApiJudge(**args)

    def test_meta_judge_parses_score(self):
        texts = ["Yes", "YES", "No", "no.", "Yes!", "No", "Yes",
                 "Yes", "No", "Yes"]
        seen = []

        def fake_urlopen(req, timeout=None):
            seen.append(req)
            return _Resp(_payload(texts[len(seen) - 1]))

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            score = self._judge().score("q?", "r.", "blunt")
        self.assertAlmostEqual(score, 0.6)
        self.assertEqual(len(seen), 10)
        body = json.loads(seen[0].data.decode("utf-8"))
        self.assertEqual(body["model"], "fake-model")
        self.assertEqual(body["temperature"], 1.0)
        self.assertIn("Yes or No", body["input"])
        self.assertEqual(seen[0].get_header("Authorization"),
                         "Bearer fake-key")

    def test_refusal_scores_zero_without_calls(self):
        with mock.patch("urllib.request.urlopen") as m:
            score = self._judge().score("q?", "Sorry, I can't help.",
                                        "blunt")
        self.assertEqual(score, 0.0)
        m.assert_not_called()

    def test_abstains_count_half(self):
        with mock.patch("urllib.request.urlopen",
                        lambda req, timeout=None: _Resp(_payload("Maybe."))):
            score = self._judge(n_samples=4).score("q?", "r.", "blunt")
        self.assertAlmostEqual(score, 0.5)

    def test_missing_key_raises_cleanly(self):
        with mock.patch.dict("os.environ", {"META_API_KEY": ""}):
            with self.assertRaises(T.MetaApiError):
                T.MetaApiJudge()

    def test_env_provides_key_and_model(self):
        with mock.patch.dict("os.environ", {"META_API_KEY": "env-key",
                                            "LOBOT_JUDGE_MODEL": "env-model"}):
            j = T.MetaApiJudge()
        self.assertEqual(j.model, "env-model")

    def test_usage_tallies(self):
        with mock.patch("urllib.request.urlopen",
                        lambda req, timeout=None: _Resp(_payload("Yes"))):
            j = self._judge(n_samples=3)
            j.score("q?", "r.", "blunt")
        self.assertEqual(j.usage, {"input_tokens": 129,
                                   "output_tokens": 570, "calls": 3})


class MetaJudgeRetryTest(unittest.TestCase):
    def test_meta_judge_retries_then_fails_cleanly(self):
        sleeps = []
        with mock.patch("urllib.request.urlopen",
                        side_effect=[_http_error(429), _http_error(500),
                                     _http_error(500), _http_error(500)]), \
                mock.patch("time.sleep", sleeps.append):
            with self.assertRaises(T.MetaApiError) as ctx:
                T.MetaApiJudge(key="fake-key", model="m",
                               max_retries=3).score("q?", "r.", "blunt")
        self.assertEqual(len(sleeps), 3)
        self.assertNotIn("fake-key", str(ctx.exception))

    def test_retry_then_success(self):
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(1)
            if len(calls) < 3:
                raise _http_error(429)
            return _Resp(_payload("Yes"))

        with mock.patch("urllib.request.urlopen", fake_urlopen), \
                mock.patch("time.sleep") as sl:
            score = T.MetaApiJudge(key="fake-key", model="m",
                                   n_samples=1).score("q?", "r.", "blunt")
        self.assertEqual(score, 1.0)
        self.assertEqual(sl.call_count, 2)

    def test_client_error_fails_fast(self):
        with mock.patch("urllib.request.urlopen",
                        side_effect=_http_error(400)), \
                mock.patch("time.sleep") as sl:
            with self.assertRaises(T.MetaApiError):
                T.MetaApiJudge(key="fake-key", model="m",
                               n_samples=1).score("q?", "r.", "blunt")
        sl.assert_not_called()

    def test_no_key_in_logs_or_report(self):
        with mock.patch("urllib.request.urlopen",
                        side_effect=_http_error(500)), \
                mock.patch("time.sleep"):
            j = T.MetaApiJudge(key="fake-secret-9", model="m",
                               max_retries=0, n_samples=1)
            with self.assertRaises(T.MetaApiError) as ctx:
                j.score("q?", "r.", "blunt")
        report = json.dumps(T.judge_info("meta_api", j.model, 0.5))
        for text in (str(ctx.exception), repr(j), report):
            self.assertNotIn("fake-secret-9", text)


class ValidateScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.V = _load_validate_script()

    def test_report_records_judge_kind_and_model(self):
        pos = {p[1] for p in T.spike.BEHAVIORS["hostility"]["probe"]}

        class FakeJudge:
            model = "fake-model"
            usage = {"input_tokens": 1, "output_tokens": 2,
                     "calls": 120}

            def score(self, q, r, rubric):
                return 0.9 if r in pos else 0.1

        with tempfile.TemporaryDirectory() as tmp:
            argv = ["validate_judge.py", "--trait", "hostility",
                   "--judge", "meta_api", "--workdir", tmp]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(T, "MetaApiJudge", FakeJudge), \
                    mock.patch("trait_store.TraitStore") as store_cls:
                rc = self.V.main()
            report = json.loads(Path(tmp, "report.json").read_text())
        self.assertEqual(rc, 0)
        self.assertEqual(report["judge"],
                         {"kind": "meta_api", "model": "fake-model",
                          "agreement": 1.0})
        self.assertEqual(report["usage"],
                         {"input_tokens": 1, "output_tokens": 2,
                          "calls": 120})
        store_cls.return_value.mark_validated.assert_not_called()

    def test_local_report_records_judge_and_validates(self):
        eng = mock.MagicMock()
        eng.revision = "abc123def456789"
        eng.fingerprint.return_value = "fp"
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["validate_judge.py", "--trait", "hostility",
                   "--workdir", tmp]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch("engine.Engine",
                               return_value=eng), \
                    mock.patch.object(T, "judge_score",
                                       side_effect=[0.9, 0.1] * 6), \
                    mock.patch("trait_store.TraitStore") as store_cls:
                rc = self.V.main()
            report = json.loads(Path(tmp, "report.json").read_text())
        self.assertEqual(rc, 0)
        self.assertEqual(report["judge"]["kind"], "local")
        self.assertEqual(report["judge"]["agreement"], 1.0)
        store_cls.return_value.mark_validated.assert_called_once()
        eng.unload.assert_called_once()


if __name__ == "__main__":
    unittest.main()
