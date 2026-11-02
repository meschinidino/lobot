"""MOD phase 1: settings in one place, doc stays in sync."""
import dataclasses
import re
import unittest
from pathlib import Path

from model_lobotomy import config

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "reference" / "configuration.md"


class SettingsTest(unittest.TestCase):
    def test_defaults(self):
        s = config.load_settings({})
        self.assertEqual(s.port, 7860)
        self.assertEqual(s.backend, "torch")
        self.assertEqual(s.llama_quant, "f16")
        self.assertEqual(s.create_lane, "torch")
        self.assertEqual(s.gpu_slots, 1)
        self.assertFalse(s.replay)
        self.assertFalse(s.no_model)
        self.assertEqual(s.meta_api_key, "")
        self.assertTrue(str(s.lobot_home).endswith(".lobot"))

    def test_overrides_match_old_reads(self):
        s = config.load_settings({
            "PORT": "9000", "LOBOT_BACKEND": "llama",
            "LOBOT_GPU_SLOTS": "0", "LOBOT_GPU_WAIT": "1",
            "MODEL_LOBOTOMY_REPLAY": "1", "LOBOT_GUARD_SCREENS": "refusal",
            "META_API_KEY": "k", "LOBOT_HOME": "/tmp/x"})
        self.assertEqual((s.port, s.backend, s.gpu_slots), (9000, "llama", 0))
        self.assertTrue(s.gpu_wait and s.replay)
        self.assertEqual(s.guard_screens, "refusal")
        self.assertEqual(str(s.lobot_home), "/tmp/x")

    def test_bad_gpu_slots_falls_back_to_one(self):
        self.assertEqual(config.load_settings(
            {"LOBOT_GPU_SLOTS": "bogus"}).gpu_slots, 1)

    def test_frozen(self):
        s = config.load_settings({})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            s.port = 1  # type: ignore[misc]

    def test_get_settings_reads_current_env(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"LOBOT_BACKEND": "llama"}):
            self.assertEqual(config.get_settings().backend, "llama")

    def test_doc_lists_every_var_and_no_extra(self):
        text = DOC.read_text()
        documented = set(re.findall(r"`([A-Z][A-Z0-9_]+)`", text))
        known = {name for name, _ in config.ENV_VARS}
        self.assertEqual(documented & known, known,
                         f"missing from doc: {sorted(known - documented)}")
        # Every documented LOBOT_/MODEL_/META_/PORT name is known.
        extra = {n for n in documented
                 if n.startswith(("LOBOT_", "MODEL_LOBOTOMY", "META_"))
                 or n == "PORT"}
        self.assertEqual(extra, known, f"doc-only vars: {sorted(extra - known)}")

    def test_no_env_reads_outside_config_in_src(self):
        src = ROOT / "src" / "model_lobotomy"
        bad = []
        for p in sorted(src.glob("*.py")):
            if p.name == "config.py":
                continue
            if re.search(r"os\.environ|os\.getenv", p.read_text()):
                bad.append(p.name)
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()


class EnvFileTest(unittest.TestCase):
    """The server reads .env at start; the shell wins; the key never reaches /api/state."""

    def test_reads_env_file_and_keeps_shell_values(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".env"
            p.write_text('# comment\nMETA_API_KEY="k-123"\nexport LOBOT_CREATE_LANE=llama\nPORT=9999\nbad line\n')
            env = {"PORT": "7860"}
            done = config.load_env_file(p, env)
        self.assertEqual(env, {"PORT": "7860", "META_API_KEY": "k-123", "LOBOT_CREATE_LANE": "llama"})
        self.assertEqual(sorted(done), ["LOBOT_CREATE_LANE", "META_API_KEY"])

    def test_missing_file_is_fine(self):
        self.assertEqual(config.load_env_file(Path("/nonexistent/.env"), {}), [])

    def test_env_file_is_git_ignored(self):
        self.assertIn(".env", (ROOT / ".gitignore").read_text().split())
