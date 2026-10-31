"""Unit tests: re-verify anywhere (RV).

No model. Proves: the library returns the script's report shape
with a scripted engine; the route streams progress then one
result; missing patches are 422 and model pins are 409; uploads
reject non-patches and oversize files; the CI workflow parses.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import patch as patch_mod  # noqa: E402
from model_lobotomy import reverify as R  # noqa: E402
from model_lobotomy import server  # noqa: E402
from tests.test_traits_api import _write_job  # noqa: E402


def _knob(path, trait="blunt", mean=0.6, ci=(0.5, 0.7),
          model="fake/model", seed=0):
    return patch_mod.save_patch(
        path, direction=torch.randn(16) * 0.5, layer=7, dose=1.5,
        trait=trait, scale=2.5, recipe={"seed": seed},
        report_card={"passed": True, "trait_mean": mean,
                     "trait_ci": list(ci)},
        guard={"blocked": False},
        fingerprint={"model_id": model, "revision": "r"})


class ScriptedEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = object()
        self.tokenizer = object()
        self.model_id = server.engine._active_model_id
        self.revision = "stub-rev"

    def load(self, model_id):
        self.model_id = model_id
        return 1

    def run(self, prompt, edits, medians=None, max_new_tokens=60,
            seed=0, **kw):
        if getattr(edits, "vector", ()):
            return {"text": "steered reply"}
        return {"text": "base reply"}


class LibraryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_library_returns_script_report(self):
        knob = _knob(self.dir / "k.lobot")
        judge = lambda q, r: 0.8 if r == "steered reply" else 0.2
        seen = []
        code, out = R.run_verify_patch(
            knob, quick=True, engine=ScriptedEngine(),
            judge_fn_for_fn=lambda eng, trait: judge,
            progress=lambda d, t: seen.append((d, t)))
        self.assertEqual(code, 0)
        self.assertTrue(out["pass"])
        self.assertEqual(out["n"], 16)
        self.assertAlmostEqual(out["measured_mean"], 0.6)
        self.assertEqual(out["trait"], "blunt")
        self.assertEqual(len(seen), 16)
        self.assertEqual(seen[-1], (16, 16))

    def test_library_full_split(self):
        knob = _knob(self.dir / "k.lobot")
        judge = lambda q, r: 0.8 if r == "steered reply" else 0.2
        code, out = R.run_verify_patch(
            knob, quick=False, engine=ScriptedEngine(),
            judge_fn_for_fn=lambda eng, trait: judge)
        self.assertEqual(code, 0)
        self.assertEqual(out["n"], 30)

    def test_library_bytes_input(self):
        knob = _knob(self.dir / "k.lobot")
        judge = lambda q, r: 0.8 if r == "steered reply" else 0.2
        code, out = R.run_verify_patch(
            knob.read_bytes(), quick=True, engine=ScriptedEngine(),
            judge_fn_for_fn=lambda eng, trait: judge)
        self.assertEqual(code, 0)
        self.assertEqual(out["n"], 16)

    def test_library_rejects_garbage(self):
        code, out = R.run_verify_patch(
            b"not a patch", quick=True, engine=ScriptedEngine())
        self.assertEqual(code, 2)
        self.assertIn("rejected", out["error"])


class RouteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.jobs_dir = Path(self.tmp.name) / "jobs"
        self.jobs_dir.mkdir()
        from model_lobotomy import create_job
        server._runner = create_job.CreateRunner(
            jobs_dir=self.jobs_dir)
        self.addCleanup(setattr, server, "_runner", None)
        self.eng = ScriptedEngine()
        self.patches = [
            mock.patch.object(server, "NO_MODEL", False),
            mock.patch.object(server, "get_engine",
                              return_value=self.eng),
        ]
        for p in self.patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.patches])
        self.client = TestClient(server.api)
        self.job = "blunt-0-20260101-000000-000"
        job_dir = _write_job(self.jobs_dir, self.job, trait="blunt")
        _knob(job_dir / "trait" / "knob.lobot",
              model=server.engine._active_model_id)

    def _events(self, r):
        self.assertEqual(r.status_code, 200, r.text)
        return [json.loads(line) for line in r.text.splitlines()
                if line]

    def test_verify_streams_progress_then_result(self):
        judge = lambda q, r: 0.8 if r == "steered reply" else 0.2
        with mock.patch.object(R, "judge_fn_for",
                               return_value=judge):
            r = self.client.post("/api/verify", json={
                "trait_id": self.job, "quick": True})
        evs = self._events(r)
        prog = [e for e in evs if e["stage"] == "progress"]
        self.assertEqual(len(prog), 16)
        self.assertEqual((prog[-1]["done"], prog[-1]["total"]),
                         (16, 16))
        self.assertEqual(evs[-1]["stage"], "done")
        self.assertTrue(evs[-1]["result"]["pass"])
        self.assertEqual(evs[-1]["exit_code"], 0)

    def test_verify_422_without_patch(self):
        (self.jobs_dir / self.job / "trait" / "knob.lobot").unlink()
        r = self.client.post("/api/verify", json={
            "trait_id": self.job, "quick": True})
        self.assertEqual(r.status_code, 422)

    def test_verify_404_unknown_trait(self):
        r = self.client.post("/api/verify", json={
            "trait_id": "nope-0", "quick": True})
        self.assertEqual(r.status_code, 404)

    def test_verify_model_pin_409(self):
        job_dir = self.jobs_dir / self.job
        _knob(job_dir / "trait" / "knob.lobot", model="other/model")
        r = self.client.post("/api/verify", json={
            "trait_id": self.job, "quick": True})
        self.assertEqual(r.status_code, 409)
        self.assertIn("not the active model", r.text)

    def test_verify_busy_409(self):
        self.eng.lock.acquire()
        try:
            r = self.client.post("/api/verify", json={
                "trait_id": self.job, "quick": True})
        finally:
            self.eng.lock.release()
        self.assertEqual(r.status_code, 409)

    def test_upload_round_trip(self):
        data = (self.jobs_dir / self.job / "trait"
                / "knob.lobot").read_bytes()
        judge = lambda q, r: 0.8 if r == "steered reply" else 0.2
        with mock.patch.object(R, "judge_fn_for",
                               return_value=judge):
            r = self.client.post(
                "/api/verify/upload",
                files={"file": ("k.lobot", data,
                                "application/zip")})
        evs = self._events(r)
        self.assertEqual(evs[-1]["stage"], "done")
        self.assertTrue(evs[-1]["result"]["pass"])

    def test_upload_rejects_non_patch(self):
        r = self.client.post(
            "/api/verify/upload",
            files={"file": ("x.txt", b"hello",
                            "text/plain")})
        self.assertEqual(r.status_code, 422)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("evil.pkl", b"pickle")
            zf.writestr("patch.json", "{}")
            zf.writestr("direction.safetensors", b"x")
        r = self.client.post(
            "/api/verify/upload",
            files={"file": ("evil.lobot", buf.getvalue(),
                            "application/zip")})
        self.assertEqual(r.status_code, 422)
        self.assertIn("pickle", r.text)

    def test_upload_cap_413(self):
        big = b"x" * (server.UPLOAD_BYTES_CAP + 1)
        r = self.client.post(
            "/api/verify/upload",
            files={"file": ("big.lobot", big,
                            "application/zip")})
        self.assertEqual(r.status_code, 413)


class WorkflowTest(unittest.TestCase):
    def test_workflow_file_valid_yaml(self):
        doc = yaml.safe_load(
            (ROOT / ".github" / "workflows"
             / "verify-patch.yml").read_text())
        call = doc[True]["workflow_call"]
        self.assertIn("patch_path", call["inputs"])
        self.assertIn("quick", call["inputs"])
        steps = doc["jobs"]["verify-patch"]["steps"]
        self.assertTrue(any("verify_patch.py" in str(s.get("run", ""))
                            for s in steps))


if __name__ == "__main__":
    unittest.main()
