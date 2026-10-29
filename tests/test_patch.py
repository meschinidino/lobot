"""Unit tests: S3-1 .lobot patch save/load + hostile rejection.

Proves: roundtrip preserves the direction and metadata; archives
with pickled payloads, bad versions or missing members are
rejected; a hostile pickle is never executed.
"""
from __future__ import annotations

import io
import json
import pickle
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import patch as patch_mod  # noqa: E402
from patch import PatchRejected, load_patch, save_patch  # noqa: E402


def _meta(**kw):
    d = {"recipe": {"r": 1}, "report_card": {"passed": True},
         "guard": {"blocked": False}, "fingerprint": {"rev": "abc"}}
    d.update(kw)
    return d


class PatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _path(self, name="knob.lobot"):
        return Path(self.tmp.name) / name

    def test_roundtrip(self):
        direction = torch.randn(16) * 0.5
        save_patch(self._path(), direction=direction, layer=7,
                   dose=1.5, trait="blunt", scale=2.5, **_meta())
        got = load_patch(self._path())
        self.assertTrue(torch.equal(got["direction"], direction))
        self.assertEqual(got["meta"]["trait"], "blunt")
        self.assertEqual(got["meta"]["layer"], 7)
        self.assertEqual(got["meta"]["dose"], 1.5)
        self.assertEqual(got["meta"]["scale"], 2.5)
        self.assertEqual(got["meta"]["version"],
                         patch_mod.PATCH_VERSION)

    def test_export_control_vector(self):
        import gguf
        direction = torch.randn(16) * 0.5
        save_patch(self._path(), direction=direction, layer=7,
                   dose=1.5, trait="blunt", scale=2.5, **_meta())
        out = Path(self.tmp.name) / "control_vector.gguf"
        got = patch_mod.export_control_vector(self._path(), out)
        self.assertEqual(got["strength"], 1.5 * 2.5)
        self.assertEqual(got["layer"], 7)
        self.assertIn("--control-vector-scaled", got["flags"])
        reader = gguf.GGUFReader(str(out))
        self.assertEqual(len(reader.tensors), 1)
        tensor = reader.tensors[0]
        self.assertEqual(tensor.name, "direction.7")
        back = torch.from_numpy(tensor.data.copy())
        self.assertTrue(torch.allclose(
            back, direction.to(torch.float32), atol=1e-6))

    def test_export_rejects_layer_zero(self):
        save_patch(self._path(), direction=torch.randn(16),
                   layer=0, dose=1.0, trait="blunt", scale=1.0,
                   **_meta())
        with self.assertRaises(PatchRejected):
            patch_mod.export_control_vector(
                self._path(), Path(self.tmp.name) / "cvec.gguf")

    def test_export_rejects_missing_scale(self):
        save_patch(self._path(), direction=torch.randn(16),
                   layer=7, dose=1.0, trait="blunt", **_meta())
        with self.assertRaises(PatchRejected):
            patch_mod.export_control_vector(
                self._path(), Path(self.tmp.name) / "cvec.gguf")

    def test_rejects_pickled_payload(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("patch.json", json.dumps(
                {"version": patch_mod.PATCH_VERSION}))
            zf.writestr("evil.pkl", pickle.dumps({"a": 1}))
        self._path().write_bytes(buf.getvalue())
        with self.assertRaises(PatchRejected) as ctx:
            load_patch(self._path())
        self.assertIn("pickled payload", str(ctx.exception))

    def test_hostile_pickle_never_executes(self):
        marker = Path(self.tmp.name) / "pwned.txt"

        class Evil:
            def __reduce__(self):
                import os
                return (os.system, (f"touch {marker}",))

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("patch.json", json.dumps(
                {"version": patch_mod.PATCH_VERSION}))
            zf.writestr("direction.pt", pickle.dumps(Evil()))
        self._path().write_bytes(buf.getvalue())
        with self.assertRaises(PatchRejected):
            load_patch(self._path())
        self.assertFalse(marker.exists())

    def test_rejects_bad_version(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("patch.json", json.dumps({"version": "v99"}))
            zf.writestr("direction.safetensors", b"dummy")
        self._path().write_bytes(buf.getvalue())
        with self.assertRaises(PatchRejected) as ctx:
            load_patch(self._path())
        self.assertIn("unsupported patch version", str(ctx.exception))

    def test_rejects_missing_members(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("patch.json", json.dumps(
                {"version": patch_mod.PATCH_VERSION}))
        self._path().write_bytes(buf.getvalue())
        with self.assertRaises(PatchRejected) as ctx:
            load_patch(self._path())
        self.assertIn("direction.safetensors", str(ctx.exception))

    def test_rejects_non_zip(self):
        self._path().write_bytes(b"not a zip")
        with self.assertRaises(PatchRejected):
            load_patch(self._path())


if __name__ == "__main__":
    unittest.main()
