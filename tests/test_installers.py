"""Unit tests: install everywhere (IN).

No model, no subprocess: the target binaries are stubbed. Proves:
probes parse versions and report missing binaries; install
refuses a missing model file; generate assembles the steered and
clean commands; verify compares texts and the shipped interval;
routes return 400/404/409/422 in the right cases.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import installers as I  # noqa: E402
from model_lobotomy import patch as patch_mod  # noqa: E402
from model_lobotomy import server  # noqa: E402
from model_lobotomy import trait_eval as T  # noqa: E402
from tests.test_traits_api import _write_job  # noqa: E402


def _proc(rc=0, out="", err=""):
    return SimpleNamespace(returncode=rc, stdout=out, stderr=err)


class ProbeTest(unittest.TestCase):
    def test_llama_probe_parses_version(self):
        with mock.patch.object(I, "_llama_binary",
                               return_value="/bin/llama-cli"), \
                mock.patch.object(I, "_run",
                                  return_value=_proc(
                                      0, "version: 1.2.3\n")) as run:
            got = I.probe("llamacpp")
        self.assertTrue(got["available"])
        self.assertEqual(got["version"], "version: 1.2.3")
        self.assertEqual(got["detail"], "/bin/llama-cli")
        run.assert_called_once_with(["/bin/llama-cli", "--version"],
                                    I.PROBE_TIMEOUT)

    def test_llama_probe_missing_binary(self):
        with mock.patch.object(I, "_llama_binary",
                               return_value=None):
            got = I.probe("llamacpp")
        self.assertFalse(got["available"])
        self.assertIn("llama-cli", got["detail"])

    def test_llama_probe_failing_version(self):
        with mock.patch.object(I, "_llama_binary",
                               return_value="/bin/llama-cli"), \
                mock.patch.object(I, "_run",
                                  return_value=_proc(1, "", "boom")):
            got = I.probe("llamacpp")
        self.assertFalse(got["available"])
        self.assertIn("boom", got["detail"])

    def test_ollama_probe_support_markers(self):
        calls = {"n": 0}

        def fake_run(cmd, timeout):
            calls["n"] += 1
            if cmd[-1] == "--version":
                return _proc(0, "ollama version 0.9.9\n")
            return _proc(0, "  --control-vector FILE\n")
        with mock.patch.object(I.shutil, "which",
                               return_value="/bin/ollama"), \
                mock.patch.object(I, "_run", side_effect=fake_run):
            got = I.probe("ollama")
        self.assertTrue(got["available"])
        self.assertIn("control-vector", got["detail"])
        self.assertEqual(calls["n"], 5)

    def test_ollama_probe_no_markers(self):
        def fake_run(cmd, timeout):
            if cmd[-1] == "--version":
                return _proc(0, "ollama version 0.9.9\n")
            return _proc(0, "  --verbose\n")
        with mock.patch.object(I.shutil, "which",
                               return_value="/bin/ollama"), \
                mock.patch.object(I, "_run", side_effect=fake_run):
            got = I.probe("ollama")
        self.assertFalse(got["available"])
        self.assertIn("no control-vector support", got["detail"])

    def test_ollama_probe_missing(self):
        with mock.patch.object(I.shutil, "which",
                               return_value=None):
            got = I.probe("ollama")
        self.assertFalse(got["available"])

    def test_unknown_target(self):
        self.assertFalse(I.probe("nope")["available"])


class InstallDriverTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "out"

    def test_install_refuses_missing_model(self):
        with mock.patch.object(I, "probe",
                               return_value={"target": "llamacpp",
                                             "available": True,
                                             "detail": "/bin/cli"}):
            got = I.install("llamacpp", "p.lobot", self.out,
                            "/none/missing.gguf")
        self.assertFalse(got["ok"])
        self.assertIn("/none/missing.gguf", got["detail"])

    def test_install_unavailable_target(self):
        with mock.patch.object(I, "probe",
                               return_value={"target": "llamacpp",
                                             "available": False,
                                             "detail": "nope"}):
            got = I.install("llamacpp", "p.lobot", self.out, "m.gguf")
        self.assertFalse(got["ok"])
        self.assertEqual(got["detail"], "nope")

    def test_install_writes_gguf_and_command(self):
        model = Path(self.tmp.name) / "m.gguf"
        model.write_bytes(b"gguf")
        exp = {"layer": 17, "strength": 2.5,
               "flags": ["--control-vector-scaled", "x:2.5"]}
        with mock.patch.object(I, "probe",
                               return_value={"target": "llamacpp",
                                             "available": True,
                                             "detail": "/bin/cli"}), \
                mock.patch.object(patch_mod, "export_control_vector",
                                  return_value=dict(exp)) as ex:
            got = I.install("llamacpp", "p.lobot", self.out,
                            str(model))
        self.assertTrue(got["ok"])
        ex.assert_called_once_with(
            "p.lobot", self.out / "control_vector.gguf")
        self.assertEqual(got["base_cmd"][:4],
                         ["/bin/cli", "-m", str(model.resolve()),
                          "--temp"])
        self.assertIn("--temp", got["base_cmd"])
        self.assertEqual(got["steer_flags"], exp["flags"])
        self.assertEqual(got["layer"], 17)

    def test_install_rejected_patch(self):
        model = Path(self.tmp.name) / "m.gguf"
        model.write_bytes(b"gguf")
        with mock.patch.object(I, "probe",
                               return_value={"target": "llamacpp",
                                             "available": True,
                                             "detail": "/bin/cli"}), \
                mock.patch.object(
                    patch_mod, "export_control_vector",
                    side_effect=patch_mod.PatchRejected("layer 0")):
            got = I.install("llamacpp", "p.lobot", self.out,
                            str(model))
        self.assertFalse(got["ok"])
        self.assertIn("rejected", got["detail"])


class GenerateTest(unittest.TestCase):
    RUN = {"base_cmd": ["/bin/cli", "-m", "m.gguf"],
           "steer_flags": ["--cv", "x"]}

    def test_generate_assembles_commands(self):
        seen = []

        def fake_run(cmd, timeout):
            seen.append(cmd)
            return _proc(0, "> hi\nhello there\n[ Prompt: 1 ]\n")
        with mock.patch.object(I, "_run", side_effect=fake_run):
            clean = I.generate("llamacpp", self.RUN, "hi",
                               patched=False, max_new=8)
            hot = I.generate("llamacpp", self.RUN, "hi",
                             patched=True, max_new=8)
        self.assertTrue(clean["ok"])
        self.assertEqual(clean["text"], "hello there")
        self.assertNotIn("--cv", seen[0])
        self.assertEqual(seen[1][-2:], ["--cv", "x"])
        self.assertIn("hi", seen[0])
        self.assertTrue(hot["ok"])

    def test_generate_failure_shapes(self):
        with mock.patch.object(I, "_run",
                               return_value=_proc(2, "", "bad")):
            got = I.generate("llamacpp", self.RUN, "hi")
        self.assertFalse(got["ok"])
        self.assertIn("exited 2", got["detail"])
        with mock.patch.object(
                I, "_run",
                side_effect=subprocess.TimeoutExpired("cli", 1)):
            got = I.generate("llamacpp", self.RUN, "hi")
        self.assertFalse(got["ok"])
        self.assertIn("timed out", got["detail"])
        with mock.patch.object(I, "_run",
                               return_value=_proc(0, "no echo\n")):
            got = I.generate("llamacpp", self.RUN, "hi")
        self.assertFalse(got["ok"])
        self.assertIn("prompt echo", got["detail"])

    def test_compare_rule(self):
        card = {"trait_mean": 0.3, "trait_ci": [0.2, 0.4]}
        self.assertTrue(I.compare(card, [0.35])["pass"])
        self.assertFalse(I.compare(card, [0.41])["pass"])
        self.assertEqual(I.compare(card, [0.35])["n"], 1)


class InstallRoutesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.jobs_dir = Path(self.tmp.name) / "jobs"
        self.jobs_dir.mkdir()
        from model_lobotomy import create_job
        server._runner = create_job.CreateRunner(
            jobs_dir=self.jobs_dir)
        self.addCleanup(setattr, server, "_runner", None)
        self.roots = mock.patch.object(
            I, "INSTALLS_ROOT", Path(self.tmp.name) / "installs")
        self.roots.start()
        self.addCleanup(self.roots.stop)
        self.client = TestClient(server.api)
        self.job = "pirate-0-20260101-000000-000"
        _write_job(self.jobs_dir, self.job, trait="pirate")

    def test_targets_shape(self):
        with mock.patch.object(I, "probe",
                               side_effect=lambda t: {
                                   "target": t, "available": False,
                                   "version": None, "detail": "x"}):
            r = self.client.get("/api/install/targets")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual([t["target"] for t in r.json()["targets"]],
                         ["llamacpp", "ollama"])

    def test_install_bad_target_and_trait(self):
        r = self.client.post("/api/install", json={
            "trait_id": self.job, "target": "nope",
            "model_ref": "m.gguf"})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/install", json={
            "trait_id": "missing-0", "target": "llamacpp",
            "model_ref": "m.gguf"})
        self.assertEqual(r.status_code, 404)

    def test_install_422_when_unavailable(self):
        with mock.patch.object(I, "install",
                               return_value={"ok": False,
                                             "detail": "no cli"}):
            r = self.client.post("/api/install", json={
                "trait_id": self.job, "target": "llamacpp",
                "model_ref": "m.gguf"})
        self.assertEqual(r.status_code, 422)
        self.assertIn("no cli", r.text)

    def test_install_ok_writes_record(self):
        got = {"ok": True, "detail": "wrote out",
               "base_cmd": ["cli"], "steer_flags": []}
        with mock.patch.object(I, "install",
                               return_value=dict(got)):
            r = self.client.post("/api/install", json={
                "trait_id": self.job, "target": "llamacpp",
                "model_ref": "m.gguf"})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertIn(self.job, d["install_id"])
        rec = I.load_install(d["install_id"])
        self.assertEqual(rec["trait_id"], self.job)
        self.assertEqual(rec["run"]["base_cmd"], ["cli"])

    def test_install_busy_409(self):
        server._install_lock.acquire()
        try:
            r = self.client.post("/api/install", json={
                "trait_id": self.job, "target": "llamacpp",
                "model_ref": "m.gguf"})
        finally:
            server._install_lock.release()
        self.assertEqual(r.status_code, 409)

    def test_verify_compares_texts(self):
        rec = {"install_id": "abc-llamacpp-1", "trait_id": self.job,
               "target": "llamacpp", "model_ref": "m.gguf",
               "run": {"base_cmd": ["cli"]}, "detail": "x"}
        I.save_install(dict(rec))
        gens = [{"ok": True, "text": "plain", "detail": ""},
                {"ok": True, "text": "ARR MATEY", "detail": ""}]
        with mock.patch.object(I, "generate",
                               side_effect=list(gens)) as gen:
            r = self.client.post(
                "/api/install/abc-llamacpp-1/verify",
                json={"prompt": "Say hi.", "judge": None})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d["base_text"], "plain")
        self.assertEqual(d["patched_text"], "ARR MATEY")
        self.assertTrue(d["differs"])
        self.assertIsNone(d["comparison"])
        self.assertEqual(gen.call_args_list[0].kwargs["patched"],
                         False)
        self.assertEqual(gen.call_args_list[1].kwargs["patched"],
                         True)

    def test_verify_404_and_400(self):
        r = self.client.post("/api/install/nope-1/verify", json={
            "prompt": "hi"})
        self.assertEqual(r.status_code, 404)
        rec = {"install_id": "abc-llamacpp-1", "trait_id": self.job,
               "target": "llamacpp", "model_ref": "m.gguf",
               "run": {"base_cmd": ["cli"]}, "detail": "x"}
        I.save_install(dict(rec))
        r = self.client.post("/api/install/abc-llamacpp-1/verify",
                             json={"prompt": "  "})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/install/abc-llamacpp-1/verify",
                             json={"prompt": "hi", "judge": "x"})
        self.assertEqual(r.status_code, 400)

    def test_verify_failed_run_422(self):
        rec = {"install_id": "abc-llamacpp-1", "trait_id": self.job,
               "target": "llamacpp", "model_ref": "m.gguf",
               "run": {"base_cmd": ["cli"]}, "detail": "x"}
        I.save_install(dict(rec))
        with mock.patch.object(I, "generate", return_value={
                "ok": False, "text": "", "detail": "cli died"}):
            r = self.client.post(
                "/api/install/abc-llamacpp-1/verify",
                json={"prompt": "hi"})
        self.assertEqual(r.status_code, 422)


class InstallCompareTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.jobs_dir = Path(self.tmp.name) / "jobs"
        self.jobs_dir.mkdir()
        from model_lobotomy import create_job
        server._runner = create_job.CreateRunner(
            jobs_dir=self.jobs_dir)
        self.addCleanup(setattr, server, "_runner", None)
        self.job = "pirate-0-20260101-000000-000"
        job_dir = _write_job(self.jobs_dir, self.job, trait="pirate")
        patch_mod.save_patch(
            job_dir / "trait" / "knob.lobot",
            direction=torch.ones(8), layer=14, dose=0.5,
            trait="pirate", recipe={},
            report_card={"trait_mean": 0.3,
                         "trait_ci": [0.2, 0.4]},
            guard={}, fingerprint={}, scale=2.0)
        self.eng = SimpleNamespace(
            lock=threading.Lock(), model=object(),
            tokenizer=object(),
            model_id=server.engine._active_model_id,
            revision="r")

    def test_compare_inside_interval(self):
        record = {"trait_id": self.job}
        with mock.patch.object(server, "NO_MODEL", False), \
                mock.patch.object(server, "get_engine",
                                  return_value=self.eng), \
                mock.patch.object(T, "family_cfg",
                                  return_value={"judge": "J"}), \
                mock.patch.object(T, "judge_score",
                                  side_effect=[0.1, 0.45]):
            got = server._install_compare(record, "q", "base",
                                          "patched")
        self.assertTrue(got["pass"])
        self.assertAlmostEqual(got["measured_mean"], 0.35)
        self.assertEqual(got["n"], 1)

    def test_compare_no_shipped_numbers(self):
        job_dir = self.jobs_dir / self.job
        patch_mod.save_patch(
            job_dir / "trait" / "knob.lobot",
            direction=torch.ones(8), layer=14, dose=0.5,
            trait="pirate", recipe={}, report_card={},
            guard={}, fingerprint={}, scale=2.0)
        with mock.patch.object(server, "NO_MODEL", False):
            got = server._install_compare({"trait_id": self.job},
                                          "q", "b", "p")
        self.assertIsNone(got["pass"])
        self.assertIn("no shipped numbers", got["reason"])


if __name__ == "__main__":
    unittest.main()
