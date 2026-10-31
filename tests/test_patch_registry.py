"""Signed registry (RG) and audit trail (AU): no model, temp dirs only.

Acceptance (later.md RG): a patch published by user A installs for
user B, verifies, and fails loudly when changed. AU: the log proves
it was not edited and rebuilds the active patches.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_lobotomy import audit  # noqa: E402
from model_lobotomy import patch as patch_mod  # noqa: E402
from model_lobotomy import registry as reg  # noqa: E402


def _patch(path, *, guard=None, card=None, model="meta-llama/x"):
    return patch_mod.save_patch(
        path, direction=torch.randn(16), layer=7, dose=1.5,
        trait="pirate", scale=2.5, recipe={"seed": 0, "judge": "j"},
        report_card=card if card is not None else {
            "passed": True, "trait_mean": 0.6, "trait_ci": [0.5, 0.7]},
        guard=guard if guard is not None else {
            "blocked": False, "would_block": False,
            "research_mode": False},
        fingerprint={"model_id": model, "revision": "rev1"})


class RegistryTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.t = Path(tmp.name)
        self.reg = self.t / "registry"
        self.log = self.t / "audit.jsonl"
        a = reg.generate_key(self.t / "alice")
        self.alice = reg.load_key(a["path"])
        self.alice_id = a["key_id"]
        self.patch = _patch(self.t / "knob.lobot")

    def _publish(self, **kw):
        return reg.publish(self.reg, self.patch, self.alice,
                           audit_log=self.log, **kw)

    def test_a_publishes_b_installs(self):
        entry = self._publish()
        self.assertEqual(entry["key_id"], self.alice_id)
        got = reg.install(self.reg, entry["name"], self.t / "bob",
                          trusted={self.alice_id}, audit_log=self.log)
        self.assertEqual(Path(got["path"]).read_bytes(),
                         self.patch.read_bytes())
        self.assertEqual([e["event"] for e in audit.read(self.log)],
                         ["publish", "install"])

    def test_key_file_is_private(self):
        k = reg.generate_key(self.t / "k")
        self.assertEqual(Path(k["path"]).stat().st_mode & 0o777, 0o600)

    def test_untrusted_publisher_refused(self):
        entry = self._publish()
        with self.assertRaisesRegex(reg.RegistryError, "not trusted"):
            reg.install(self.reg, entry["name"], self.t / "bob",
                        trusted=set(), audit_log=self.log)

    def test_changed_patch_bytes_fail(self):
        entry = self._publish()
        p = self.reg / "patches" / f"{entry['patch_sha256']}.lobot"
        data = bytearray(p.read_bytes())
        data[-5] ^= 0xFF
        p.write_bytes(bytes(data))
        with self.assertRaisesRegex(reg.RegistryError, "changed"):
            reg.install(self.reg, entry["name"], self.t / "bob",
                        trusted={self.alice_id}, audit_log=self.log)

    def test_changed_index_entry_fails_signature(self):
        entry = self._publish()
        idx = json.loads((self.reg / "index.json").read_text())
        idx["entries"][0]["model_id"] = "someone/else"
        (self.reg / "index.json").write_text(json.dumps(idx))
        with self.assertRaisesRegex(reg.RegistryError, "signature"):
            reg.install(self.reg, entry["name"], self.t / "bob",
                        trusted={self.alice_id}, audit_log=self.log)

    def test_swapped_patch_with_resigned_entry_fails_claims(self):
        # Mallory re-signs a different patch under Alice's name: the
        # signature check fails because Mallory's key is not Alice's.
        entry = self._publish()
        m = reg.load_key(reg.generate_key(self.t / "m")["path"])
        idx = json.loads((self.reg / "index.json").read_text())
        e = idx["entries"][0]
        e["signature"] = m.sign(reg.statement_of(e)).hex()
        (self.reg / "index.json").write_text(json.dumps(idx))
        with self.assertRaisesRegex(reg.RegistryError, "signature"):
            reg.verify_entry(self.reg, e, {self.alice_id})
        del entry

    def test_dual_use_refusals(self):
        cases = {
            "blocked": {"blocked": True, "research_mode": False},
            "would_block": {"blocked": False, "would_block": True},
            "research": {"blocked": False, "research_mode": True},
        }
        for name, guard in cases.items():
            p = _patch(self.t / f"{name}.lobot", guard=guard)
            with self.assertRaises(reg.RegistryError, msg=name):
                reg.publish(self.reg, p, self.alice, audit_log=self.log)
        empty = patch_mod.save_patch(
            self.t / "noguard.lobot", direction=torch.randn(16),
            layer=7, dose=1.0, trait="pirate", scale=2.0, recipe={},
            report_card={"trait_mean": 0.5, "trait_ci": [0.4, 0.6]},
            guard={}, fingerprint={"model_id": "m", "revision": "r"})
        with self.assertRaisesRegex(reg.RegistryError, "no guard"):
            reg.publish(self.reg, empty, self.alice, audit_log=self.log)
        p = _patch(self.t / "noev.lobot", card={"passed": True})
        with self.assertRaisesRegex(reg.RegistryError, "evidence"):
            reg.publish(self.reg, p, self.alice, audit_log=self.log)

    def test_duplicate_publish_refused(self):
        self._publish()
        with self.assertRaisesRegex(reg.RegistryError, "already"):
            self._publish(name="other")

    def test_trust_list_roundtrip(self):
        path = self.t / "trusted.json"
        self.assertEqual(reg.load_trusted(path), set())
        reg.trust(self.alice_id, path)
        self.assertEqual(reg.load_trusted(path), {self.alice_id})

    def test_find_by_sha_prefix(self):
        entry = self._publish()
        self.assertEqual(reg.find(self.reg, entry["patch_sha256"][:10]),
                         entry)


class AuditTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log = Path(tmp.name) / "audit.jsonl"

    def test_chain_ok_and_history(self):
        audit.record("install", log=self.log, patch_sha256="a",
                     path="/p/a.lobot")
        audit.record("install", log=self.log, patch_sha256="b",
                     path="/p/b.lobot")
        audit.record("verify", log=self.log, patch_sha256="a",
                     ok=True)
        audit.record("uninstall", log=self.log, patch_sha256="b",
                     path="/p/b.lobot")
        self.assertTrue(audit.verify_chain(self.log)["ok"])
        h = audit.history(self.log)
        self.assertEqual(list(h["active"]), ["/p/a.lobot"])
        self.assertEqual(len(h["active"]["/p/a.lobot"]["verifies"]), 1)

    def test_edited_line_detected(self):
        for i in range(3):
            audit.record("install", log=self.log, patch_sha256=str(i),
                         path=f"/p/{i}")
        lines = self.log.read_text().splitlines()
        e = json.loads(lines[1])
        e["patch_sha256"] = "evil"
        lines[1] = json.dumps(e, sort_keys=True)
        self.log.write_text("\n".join(lines) + "\n")
        got = audit.verify_chain(self.log)
        self.assertFalse(got["ok"])
        self.assertEqual(got["bad_seq"], 1)
        with self.assertRaises(ValueError):
            audit.history(self.log)

    def test_deleted_line_detected(self):
        for i in range(3):
            audit.record("install", log=self.log, patch_sha256=str(i),
                         path=f"/p/{i}")
        lines = self.log.read_text().splitlines()
        self.log.write_text("\n".join([lines[0], lines[2]]) + "\n")
        self.assertFalse(audit.verify_chain(self.log)["ok"])

    def test_empty_log_is_ok(self):
        self.assertEqual(audit.verify_chain(self.log)["n"], 0)


def _cli():
    spec = importlib.util.spec_from_file_location(
        "lobot_registry_cli", ROOT / "scripts" / "lobot_registry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CliTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.t = Path(tmp.name)
        env = mock.patch.dict(os.environ, {
            "LOBOT_AUDIT_LOG": str(self.t / "audit.jsonl")})
        env.start()
        self.addCleanup(env.stop)
        home = mock.patch.object(reg, "HOME", self.t / "home")
        home.start()
        self.addCleanup(home.stop)
        self.cli = _cli()

    def _run(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(io.StringIO()):
            code = self.cli.main(list(argv))
        return code, out.getvalue()

    def test_keygen_publish_trust_install_audit(self):
        code, out = self._run("keygen")
        key = json.loads(out)
        knob = _patch(self.t / "knob.lobot")
        reg_dir = str(self.t / "reg")
        self.assertEqual(self._run("publish", reg_dir, str(knob),
                                   "--key", key["path"])[0], 0)
        # Not trusted yet: install refuses with exit 1.
        self.assertEqual(self._run("install", reg_dir, "pirate@meta-llama/x",
                                   str(self.t / "dest"))[0], 1)
        self._run("trust", key["key_id"])
        code, out = self._run("install", reg_dir, "pirate@meta-llama/x",
                              str(self.t / "dest"))
        self.assertEqual(code, 0, out)
        code, out = self._run("audit")
        self.assertEqual(code, 0)
        got = json.loads(out)
        self.assertTrue(got["chain"]["ok"])
        self.assertEqual(len(got["active"]), 1)

    def test_ci_fails_when_evidence_breaks(self):
        key = reg.generate_key(self.t / "k")
        k = reg.load_key(key["path"])
        reg_dir = self.t / "reg"
        reg.publish(reg_dir, _patch(self.t / "a.lobot"), k, name="a")
        reg.publish(reg_dir, _patch(self.t / "b.lobot"), k, name="b")

        class Eng:
            backend = "llama"

            def load(self, model_id):
                return 1

        results = {"a": (0, {"measured_mean": 0.6, "shipped_ci": [.5, .7],
                             "n": 16}),
                   "b": (1, {"measured_mean": 0.1, "shipped_ci": [.5, .7],
                             "n": 16})}
        names = {e["patch_sha256"]: e["name"]
                 for e in reg.load_index(reg_dir)["entries"]}

        def fake_verify(path, quick, engine):
            return results[names[Path(path).stem]]

        from model_lobotomy import engine as engine_mod
        from model_lobotomy import reverify
        with mock.patch.object(engine_mod, "get_engine",
                               return_value=Eng()), \
                mock.patch.object(reverify, "run_verify_patch",
                                  side_effect=fake_verify):
            code, out = self._run("ci", str(reg_dir), "--quick")
        self.assertEqual(code, 1)
        rep = json.loads(out)
        self.assertEqual(rep["failed"], 1)
        self.assertEqual({r["name"]: r["ok"] for r in rep["entries"]},
                         {"a": True, "b": False})
        events = audit.read(self.t / "audit.jsonl")
        self.assertEqual([e["event"] for e in events][-2:],
                         ["verify", "verify"])


if __name__ == "__main__":
    unittest.main()
