"""Route sweep on the llama backend with a fake daemon (D5 sweep).

Every /api/* route once: 2xx or 501 only. No 500, no escaping
UnsupportedError/AttributeError. Fake daemon + real cached
tokenizers + committed calibration caches; install-verify runs the
real llama-cli on the existing gate GGUF. Labeled: fake daemon.
"""
import io
import json
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from model_lobotomy import engine as engine_mod
from model_lobotomy import llama_backend as lb
from model_lobotomy import patch as patch_mod
from model_lobotomy import server
from model_lobotomy import steer

ROOT = Path(__file__).resolve().parent.parent
QWEN = steer.MODEL_ID
LLAMA = steer.LLAMA_MODEL_ID
CLI = "/tmp/llama.cpp/build/bin/llama-cli"
GATE_GGUF = ROOT / "logs" / "gate_llamacpp" / "model-f16.gguf"

DRAFT_JSON = json.dumps({
    "pos": "Reply tersely in lowercase.",
    "neg": "Reply verbosely with flourish.",
    "judge": "terse lowercase replies",
    "baselines": ["b1", "b2", "b3", "b4", "b5", "b6"],
})


class SweepProc:
    """Fake daemon: scripted per prompt, real-size canned logits."""

    def __init__(self, tok, spec):
        self.tok = tok
        self.info = {"ok": True, "n_layer": spec.layers,
                     "n_embd": spec.hidden, "n_vocab": len(tok),
                     "n_ctx": 128, "desc": "sweep-fake",
                     "commit": "sweep-fake"}
        vec = torch.zeros(len(tok), dtype=torch.float32)
        from model_lobotomy import trait_eval as te
        y, _n = te.spike._yes_no_ids(tok)
        vec[y] = 5.0  # judge P(yes) ~= 1, deterministic
        self.logits = vec.numpy().tobytes()
        self.prefills = []
        self.closed = False

    def prefill(self, ids, cvec):
        self.prefills.append(list(ids))
        return self.logits

    def generate(self, ids, max_new, stop_ids, cvec,
                 cancel=None):
        text = self.tok.decode(ids, skip_special_tokens=True)
        if "contrast pair" in text:
            script = self.tok(
                DRAFT_JSON,
                add_special_tokens=False)["input_ids"]
        else:
            script = self.tok(
                "A calm reply for the sweep.",
                add_special_tokens=False)["input_ids"]
        for t in script[:max_new]:
            yield ("token", int(t))
        yield ("done", {"stop": "maxlen", "cancelled": False,
                        "prefill_ms": 1.0, "decode_ms": 2.0})

    def close(self):
        self.closed = True


class SweepReader:
    """DownProjReader stub: shared ones block (shape only)."""

    def __init__(self):
        self.blocks = {}

    def column_block(self, spec, layer):
        key = (spec.hidden, spec.intermediate)
        if key not in self.blocks:
            self.blocks[key] = torch.ones(*key)
        return self.blocks[key]


def _attach(model_id):
    """Warm fake LlamaEngine on the committed cache revision."""
    saved = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        tok = steer.load_tokenizer(model_id)
    finally:
        if saved is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = saved
    cache = json.loads(
        (ROOT / steer.calibration_path_for(model_id)).read_text())
    eng = lb.LlamaEngine(
        proc_factory=lambda b, g: SweepProc(
            tok, steer.get_spec(model_id)))
    eng._proc = SweepProc(tok, steer.get_spec(model_id))
    eng._sidecar = {"quant": "f16", "gguf_sha256": "sweep",
                    "llamacpp_commit": "sweep",
                    "revision": cache["key"]["revision"]}
    eng.model = lb.LlamaModelHandle(eng)
    eng.tokenizer = tok
    eng.model_id = model_id
    eng.revision = cache["key"]["revision"]
    eng.rung = 1
    eng._reader = SweepReader()
    eng._stop_ids = [tok.eos_token_id]
    return eng


def _ndjson(resp):
    return [json.loads(line) for line in
            resp.text.strip().splitlines() if line.strip()]


class SweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.client = TestClient(server.api,
                                raise_server_exceptions=False)
        cls._saved_model = server.engine._active_model_id
        cls.eng = _attach(QWEN)
        cls._patcher = mock.patch.object(
            server, "get_engine", return_value=cls.eng)
        cls._patcher.start()
        cls.tmp_installs = tempfile.TemporaryDirectory(
            prefix="sweep-installs-")
        cls._patcher2 = mock.patch.object(
            server.install_mod, "INSTALLS_ROOT",
            Path(cls.tmp_installs.name))
        cls._patcher2.start()
        cls.saved_cli = os.environ.get("LOBOT_LLAMACPP_CLI")
        os.environ["LOBOT_LLAMACPP_CLI"] = CLI
        # Temp family store (seeded copy): validate-agree must not
        # touch the user's committed store/traits.json.
        from model_lobotomy import trait_store as ts_mod
        cls._traits_src = ts_mod.DEFAULT_PATH
        cls._traits_sha = cls._sha(cls._traits_src)
        cls.tmp_store = tempfile.TemporaryDirectory(
            prefix="sweep-store-")
        shutil.copy(cls._traits_src,
                    Path(cls.tmp_store.name) / "traits.json")
        cls._patcher3 = mock.patch.object(
            ts_mod, "DEFAULT_PATH",
            Path(cls.tmp_store.name) / "traits.json")
        cls._patcher3.start()
        cls._saved_runner = server._runner
        server._runner = None  # rebuild against the temp store
        cls.made_jobs = []
        cls.made_drafts = []

    @staticmethod
    def _sha(path):
        import hashlib
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        cls._patcher.stop()
        cls._patcher2.stop()
        cls._patcher3.stop()
        server._runner = cls._saved_runner
        if cls.saved_cli is None:
            os.environ.pop("LOBOT_LLAMACPP_CLI", None)
        else:
            os.environ["LOBOT_LLAMACPP_CLI"] = cls.saved_cli
        server.engine._active_model_id = cls._saved_model
        assert cls._sha(cls._traits_src) == cls._traits_sha, \
            "sweep changed store/traits.json"
        cls.tmp_installs.cleanup()
        cls.tmp_store.cleanup()
        for jid in cls.made_jobs:
            shutil.rmtree(ROOT / "jobs" / jid,
                          ignore_errors=True)
        for did in cls.made_drafts:
            shutil.rmtree(ROOT / "store" / "drafts" / did,
                          ignore_errors=True)

    @classmethod
    def _job(cls, jid, trait, layer, edit_kind=None,
             report_card=None, dims=2560):
        d = ROOT / "jobs" / jid
        (d / "trait").mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(json.dumps(
            {"id": jid, "status": "done"}))
        (d / "trait" / "report.json").write_text(json.dumps(
            {"trait": trait}))
        (d / "trait" / "report.md").write_text(f"# {trait}\n")
        direction = torch.ones(dims) / float(dims)
        patch_mod.save_patch(
            d / "trait" / "knob.lobot", direction=direction,
            layer=layer, dose=1.0, trait=trait, recipe={},
            report_card=(report_card if report_card is not None
                         else {}),
            guard={}, fingerprint={"model_id": QWEN},
            scale=20.0)
        if edit_kind is not None:
            zp = d / "trait" / "knob.lobot"
            with zipfile.ZipFile(zp) as zf:
                members = {n: zf.read(n) for n in zf.namelist()}
            meta = json.loads(members["patch.json"])
            meta["edit_kind"] = edit_kind
            members["patch.json"] = json.dumps(
                meta, indent=2).encode()
            zp.unlink()
            with zipfile.ZipFile(zp, "w",
                                 zipfile.ZIP_DEFLATED) as zf:
                for n, b in members.items():
                    zf.writestr(n, b)
        cls.made_jobs.append(jid)
        return jid

    def test_01_state(self):
        r = self.client.get("/api/state")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["backend"], "llama")
        self.assertEqual(sorted(body["torch_only"]),
                         sorted(engine_mod.FEATURES))

    def test_02_run_plain(self):
        r = self.client.post("/api/run", json={
            "prompt": "Say hello.",
            "rows": [{"layer": 13, "neuron": 0, "dose": 1.0}],
            "guard": False, "research_mode": True})
        self.assertEqual(r.status_code, 200)
        stages = [e.get("stage") for e in _ndjson(r)]
        self.assertIn("complete", stages)

    def test_03_run_dial_501(self):
        r = self.client.post("/api/run", json={
            "prompt": "Say hello.",
            "rows": [{"layer": 13, "neuron": 0, "dose": 1.0}],
            "dial_trait": "cheerful"})
        self.assertEqual(r.status_code, 501)
        self.assertIn("dial needs LOBOT_BACKEND=torch",
                      r.json()["detail"])

    def test_04_stop_families_jobs(self):
        self.assertEqual(
            self.client.post("/api/stop").status_code, 200)
        self.assertEqual(
            self.client.get("/api/families").status_code, 200)
        r = self.client.get("/api/jobs")
        self.assertEqual(r.status_code, 200)
        self.assertIn("jobs", r.json())

    def test_05_create_gated_status_live(self):
        r = self.client.post("/api/create", json={
            "description": "speaks in haiku"})
        self.assertEqual(r.status_code, 501)
        jid = self._job("sweep-create-000", "cheerful", 13)
        r = self.client.get(f"/api/create/{jid}")
        self.assertEqual(r.status_code, 200)
        runner = server._get_runner()
        runner._jobs["sweep-live-000"] = SimpleNamespace(
            status="done",
            to_dict=lambda: {"id": "sweep-live-000",
                             "status": "done"})
        try:
            r = self.client.post(
                "/api/create/sweep-live-000/stop")
            self.assertEqual(r.status_code, 200)
        finally:
            runner._jobs.pop("sweep-live-000", None)

    def _draft(self, description):
        r = self.client.post("/api/drafts", json={
            "description": description, "seed": 0})
        self.assertEqual(r.status_code, 200, r.text[:300])
        did = r.json()["id"]
        self.made_drafts.append(did)
        r = self.client.get(f"/api/drafts/{did}")
        self.assertEqual(r.status_code, 200)
        return did, r.json()["replies"]

    def test_06_drafts_disagree_422(self):
        did, replies = self._draft("speaks tersely in lowercase")
        self.assertEqual(len(replies), 20)
        rows = [{"reply_id": p["reply_id"], "label": False}
                for p in replies]
        r = self.client.post(f"/api/drafts/{did}/labels",
                             json={"labels": rows})
        self.assertEqual(r.status_code, 200, r.text[:300])
        n_before = len(self.eng._proc.prefills)
        r = self.client.post(f"/api/drafts/{did}/validate")
        self.assertEqual(r.status_code, 422, r.text[:300])
        body = r.json()["detail"]
        self.assertFalse(body["registered"])
        self.assertEqual(body["agreement"], 0.0)
        self.assertGreater(len(self.eng._proc.prefills),
                           n_before)

    def test_06b_drafts_agree_200(self):
        did, replies = self._draft("speaks tersely and agrees")
        rows = [{"reply_id": p["reply_id"], "label": True}
                for p in replies]
        r = self.client.post(f"/api/drafts/{did}/labels",
                             json={"labels": rows})
        self.assertEqual(r.status_code, 200, r.text[:300])
        r = self.client.post(f"/api/drafts/{did}/validate")
        self.assertEqual(r.status_code, 200, r.text[:300])
        self.assertTrue(r.json()["registered"])
        from model_lobotomy import trait_store as ts_mod
        tmp_fams = json.loads(
            ts_mod.DEFAULT_PATH.read_text())["families"]
        self.assertIn(did, [f["name"] for f in tmp_fams])

    def test_07_traits(self):
        jid = self._job("sweep-trait-000", "cheerful", 13)
        for path in (f"/api/traits/{jid}/report",
                     f"/api/traits/{jid}/report.md",
                     f"/api/traits/{jid}/patch",
                     "/api/traits"):
            r = self.client.get(path)
            self.assertEqual(r.status_code, 200, path)

    def test_08_update_explicit(self):
        r = self.client.post("/api/update", json={
            "prompt": "Say hello.",
            "layer": 13,
            "direction": [0.01] * 2560,
            "scale": 20.0,
            "doses": [0.0, 1.0],
            "max_new_tokens": 8,
            "guard": False, "research_mode": True})
        self.assertEqual(r.status_code, 200, r.text[:300])
        stages = [e.get("stage") for e in _ndjson(r)]
        self.assertIn("complete", stages)

    def test_09_delete_gated(self):
        self._job("sweep-del-000", "cheerful", 13)
        for path, body in (
                ("/api/delete", {"prompt": "hi",
                                 "trait": "cheerful"}),
                ("/api/delete/check", {"trait": "cheerful"})):
            r = self.client.post(path, json=body)
            self.assertEqual(r.status_code, 501, path)
            self.assertIn("ablate needs LOBOT_BACKEND=torch",
                          r.json()["detail"])

    def test_10_install_roundtrip(self):
        if not (Path(CLI).is_file() and GATE_GGUF.is_file()):
            self.skipTest("needs llama-cli + gate GGUF")
        r = self.client.get("/api/install/targets")
        self.assertEqual(r.status_code, 200)
        jid = self._job(
            "sweep-install-000", "cheerful", 13,
            report_card={"trait_mean": 0.5,
                         "trait_ci": [0.4, 0.6]},
            dims=3072)
        r = self.client.post("/api/install", json={
            "trait_id": jid, "target": "llamacpp",
            "model_ref": str(GATE_GGUF)})
        self.assertEqual(r.status_code, 200, r.text[:300])
        iid = r.json()["install_id"]
        r = self.client.post(f"/api/install/{iid}/verify",
                             json={"prompt": "Say hello.",
                                   "max_new_tokens": 8})
        self.assertEqual(r.status_code, 200, r.text[:500])
        r = self.client.post(f"/api/install/{iid}/verify",
                             json={"prompt": "Say hello.",
                                   "max_new_tokens": 8,
                                   "judge": "local"})
        self.assertEqual(r.status_code, 200, r.text[:500])
        self.assertIsNotNone(r.json()["comparison"])

    def test_11_verify_gated(self):
        jid = self._job("sweep-verify-000", "cheerful", 13)
        r = self.client.post("/api/verify",
                             json={"trait_id": jid})
        self.assertEqual(r.status_code, 501)
        knob = (ROOT / "jobs" / jid / "trait" / "knob.lobot")
        with open(knob, "rb") as f:
            r = self.client.post(
                "/api/verify/upload",
                files={"file": ("knob.lobot", f,
                                "application/octet-stream")})
        self.assertEqual(r.status_code, 501)

    def test_12_combine(self):
        a = self._job("sweep-comb-a", "cheerful", 13)
        b = self._job("sweep-comb-b", "cheerful", 14)
        r = self.client.post("/api/combine/check", json={
            "patch_ids": [a, b], "max_new_tokens": 4,
            "seed": 0})
        self.assertEqual(r.status_code, 200, r.text[:400])
        c = self._job("sweep-comb-c", "cheerful", 13,
                      edit_kind="ablate")
        r = self.client.post("/api/combine/check", json={
            "patch_ids": [a, c], "max_new_tokens": 4,
            "seed": 0})
        self.assertEqual(r.status_code, 501)
        self.assertIn("ablate needs LOBOT_BACKEND=torch",
                      r.json()["detail"])

    def test_13_capture_reader_sae_gated(self):
        r = self.client.post("/api/neurons", json={
            "prompt": "Say hello.", "token_index": 1})
        self.assertEqual(r.status_code, 501)
        self.assertIn("capture needs", r.json()["detail"])
        r = self.client.get("/api/neuron/13/0")
        self.assertEqual(r.status_code, 501)
        r = self.client.get("/api/find",
                            params={"word": "hello"})
        self.assertEqual(r.status_code, 501)
        r = self.client.post("/api/lift", json={
            "layer": 13, "neuron": 0, "dose": 1.0,
            "words": ["hello"], "prompts": ["Say hello."]})
        self.assertEqual(r.status_code, 501)
        r = self.client.get("/api/features")
        self.assertEqual(r.status_code, 200)
        # Feature routes check for a trained SAE before the gate;
        # run them with Llama active (L12 SAE exists on disk).
        server.engine._active_model_id = LLAMA
        llama_eng = _attach(LLAMA)
        with mock.patch.object(server, "get_engine",
                               return_value=llama_eng):
            r = self.client.get("/api/feature/12/0")
            self.assertEqual(r.status_code, 501, r.text[:200])
            r = self.client.post("/api/feature-lift", json={
                "layer": 12, "index": 0, "dose": 1.0,
                "words": ["hello"],
                "prompts": ["Say hello."]})
            self.assertEqual(r.status_code, 501, r.text[:200])
        server.engine._active_model_id = QWEN

    def test_14_chat(self):
        r = self.client.post("/api/chat/start", json={
            "rows": [{"layer": 13, "neuron": 0,
                      "dose": 1.0}]})
        self.assertEqual(r.status_code, 200, r.text[:300])
        sid = r.json()["session_id"]
        r = self.client.post("/api/chat", json={
            "session_id": sid, "message": "hi there"})
        self.assertEqual(r.status_code, 200, r.text[:300])
        reply = None
        for e in _ndjson(r):
            if e.get("stage") == "done":
                reply = e["result"]["reply"]
        self.assertIsNotNone(reply)
        r = self.client.get(f"/api/chat/{sid}")
        self.assertEqual(r.status_code, 200)
        r = self.client.get(f"/api/chat/{sid}/transcript")
        self.assertEqual(r.status_code, 200)
        r = self.client.post("/api/chat/replay", json={
            "transcript": {
                "model_id": QWEN,
                "turns": [{"message": "hi there", "rows": [],
                           "reply": reply, "dials": {},
                           "max_new_tokens": 32}]}})
        self.assertEqual(r.status_code, 200, r.text[:400])
        r = self.client.post("/api/chat/start", json={
            "dial_traits": ["cheerful"]})
        self.assertEqual(r.status_code, 501)

    def test_15_combine_installed_and_model(self):
        # Non-empty installed matrix == _combine_matrix (covered in
        # test_12); clear the tmp installs so this hits the empty
        # branch deterministically (the install-test knob is
        # Llama-shaped and must not leak into a Qwen matrix).
        for p in Path(self.tmp_installs.name).iterdir():
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink()
        r = self.client.get("/api/combine/installed")
        self.assertEqual(r.status_code, 200, r.text[:400])
        self.assertEqual(r.json()["matrix"], None)
        r = self.client.post("/api/model",
                             json={"model_id": LLAMA})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["active_model"], LLAMA)
        r = self.client.post("/api/model", json={"model_id": QWEN})
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
