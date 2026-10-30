"""Unit tests: P1 patch install and re-verify (no weights).

install_patch.py and verify_patch.py live in scripts/ (not a
package), so they load from the file path. The verify check runs
against a fake engine and a fake judge; the model-backed round
trip stays in the model slot.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import patch as patch_mod  # noqa: E402

_SPEC_INSTALL = importlib.util.spec_from_file_location(
    "install_patch", ROOT / "scripts" / "install_patch.py")
install_mod = importlib.util.module_from_spec(_SPEC_INSTALL)
_SPEC_INSTALL.loader.exec_module(install_mod)

_SPEC_VERIFY = importlib.util.spec_from_file_location(
    "verify_patch", ROOT / "scripts" / "verify_patch.py")
verify_mod = importlib.util.module_from_spec(_SPEC_VERIFY)
_SPEC_VERIFY.loader.exec_module(verify_mod)


def _save_knob(path, **kw):
    d = {"direction": torch.randn(16) * 0.5, "layer": 7,
         "dose": 1.5, "trait": "blunt", "scale": 2.5,
         "recipe": {"seed": 0}, "report_card": {"passed": True},
         "guard": {"blocked": False}, "fingerprint": {"rev": "abc"}}
    d.update(kw)
    return patch_mod.save_patch(path, **d)


class FakeEngine:
    """Fake engine: base and steered runs return fixed texts."""

    def __init__(self):
        self.models = []

    def run(self, prompt, edits, medians=None, max_new_tokens=60,
            seed=0, **kw):
        _ = (prompt, medians, max_new_tokens, seed, kw)
        if getattr(edits, "vector", ()):
            return {"text": "steered reply"}
        return {"text": "base reply"}

    def unload(self):
        pass


class InstallTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_install_llamacpp_writes_gguf_and_flags(self):
        import gguf
        knob = _save_knob(self.dir / "knob.lobot")
        out = self.dir / "installed"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = install_mod.main(
                [str(knob), "--target", "llamacpp", "--out", str(out)])
        text = buf.getvalue()
        self.assertEqual(code, 0)
        gguf_path = out / "control_vector.gguf"
        self.assertTrue(gguf_path.exists())
        reader = gguf.GGUFReader(str(gguf_path))
        self.assertEqual([t.name for t in reader.tensors],
                         ["direction.7"])
        self.assertIn("--control-vector-scaled", text)
        self.assertIn(f"{gguf_path}:3.75", text)
        self.assertIn("--control-vector-layer-range 7 7", text)

    def test_install_ollama_unsupported_exits_2(self):
        knob = _save_knob(self.dir / "knob.lobot")
        out = self.dir / "installed"
        with mock.patch.dict(os.environ, {"PATH": str(self.dir)}):
            self.assertIsNone(__import__("shutil").which("ollama"))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = install_mod.main(
                    [str(knob), "--target", "ollama",
                     "--out", str(out)])
        self.assertEqual(code, 2)
        self.assertIn("not installed", buf.getvalue())
        self.assertFalse((out / "Modelfile.lobot").exists())

    def test_install_ollama_supported_writes_snippet(self):
        knob = _save_knob(self.dir / "knob.lobot")
        fake = self.dir / "ollama"
        fake.write_text("#!/bin/sh\n"
                        "echo 'ollama version 9.9.9 (fake)'\n"
                        "echo '--control-vector-scaled FILE:SCALE'\n")
        fake.chmod(0o755)
        out = self.dir / "installed"
        code, detail = install_mod.install_ollama(
            knob, out, ollama_exe=str(fake))
        self.assertEqual(code, 0, detail)
        snippet = (out / "Modelfile.lobot").read_text()
        self.assertIn("control_vector.gguf", snippet)
        self.assertTrue((out / "control_vector.gguf").exists())


class VerifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _knob(self, trait_mean=0.6, trait_ci=(0.5, 0.7)):
        return _save_knob(
            self.dir / "knob.lobot",
            report_card={"passed": True, "trait_mean": trait_mean,
                         "trait_ci": list(trait_ci)},
            fingerprint={"model_id": "fake/model", "revision": "r"})

    def _run(self, knob, judge):
        eng = FakeEngine()
        with mock.patch.object(verify_mod, "load_engine",
                               lambda model_id: eng), \
                mock.patch.object(verify_mod, "judge_fn_for",
                                  lambda eng, trait: judge):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = verify_mod.main([str(knob), "--quick"])
        return code, buf.getvalue(), eng

    def test_verify_quick_compares_numbers(self):
        knob = self._knob()
        judge = lambda q, r: 0.8 if r == "steered reply" else 0.2
        code, text, _ = self._run(knob, judge)
        self.assertEqual(code, 0)
        self.assertIn("prompts: 16 (quick)", text)
        self.assertIn("shipped effect 0.600", text)
        self.assertIn("measured effect 0.600", text)
        self.assertIn("PASS", text)

    def test_verify_quick_outside_interval_fails(self):
        knob = self._knob()
        judge = lambda q, r: 0.25 if r == "steered reply" else 0.2
        code, text, _ = self._run(knob, judge)
        self.assertEqual(code, 1)
        self.assertIn("measured effect 0.050", text)
        self.assertIn("FAIL", text)


if __name__ == "__main__":
    unittest.main()
