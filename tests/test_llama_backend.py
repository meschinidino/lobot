"""Llama backend unit tests: fake daemon only, no weights, no binary.

Everything here runs against FakeProc (same methods as DaemonProc)
and stub weights. Labeled accordingly: these prove the protocol
plumbing and the edit math, not end-to-end parity (that is the
model-backed lane in docs/plans/llama-backend.md, D5 parts 2-5).
"""
import base64
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from model_lobotomy import engine as engine_mod
from model_lobotomy import llama_backend as lb
from model_lobotomy import steer


TINY = steer.ModelSpec("fake-tiny", 4, 8, 16, "llama", False)


class FakeTok:
    """Tokenizer stub: fixed ids in, word-per-id text out."""

    def __init__(self, vocab=64):
        self._vocab = vocab
        self.eos_token_id = 1
        self.chat_template = "fake-template"

    def __len__(self):
        return self._vocab

    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[5, 6, 7]])}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{i}" for i in list(ids))

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        return "wrapped:" + messages[0]["content"]


class FakeReader:
    """DownProjReader stub: one fixed [H, I] block per layer."""

    def __init__(self, spec=TINY, seed=0):
        self._spec = spec
        self._seed = seed
        self.block = None
        self.calls = []

    def column_block(self, spec, layer):
        self.calls.append(layer)
        if self.block is None:
            g = torch.Generator().manual_seed(self._seed)
            self.block = torch.randn(
                self._spec.hidden, self._spec.intermediate,
                dtype=torch.float32, generator=g)
        return self.block


class FakeProc:
    """Fake lobot-llamad: records requests, scripts replies."""

    def __init__(self, binary=None, gguf=None, *, vocab=64,
                 layers=4, hidden=8, script=(11, 12, 13),
                 top_logit=7):
        _ = (binary, gguf)
        self.info = {"ok": True, "n_layer": layers,
                     "n_embd": hidden, "n_vocab": vocab,
                     "n_ctx": 128, "desc": "fake",
                     "commit": "fake"}
        self.script = list(script)
        vec = torch.zeros(vocab, dtype=torch.float32)
        vec[top_logit % vocab] = 5.0
        self.logits = vec.numpy().tobytes()
        self.prefills = []
        self.generates = []
        self.stream_closed = False
        self.closed = False

    def prefill(self, ids, cvec):
        self.prefills.append((list(ids), cvec))
        return self.logits

    def generate(self, ids, max_new, stop_ids, cvec,
                 cancel=None):
        self.generates.append(
            {"ids": list(ids), "max_new": max_new,
             "stop_ids": list(stop_ids), "cvec": cvec is not None,
             "il": None if cvec is None else (cvec[1], cvec[2])})
        try:
            for t in self.script[:max_new]:
                if cancel is not None and cancel.is_set():
                    break
                yield ("token", t)
            yield ("done", {"stop": "maxlen", "cancelled": bool(
                cancel is not None and cancel.is_set()),
                "prefill_ms": 1.0, "decode_ms": 2.0})
        finally:
            self.stream_closed = True

    def close(self):
        self.closed = True


def _llama(proc=None, tok=None, reader=None):
    eng = lb.LlamaEngine(
        proc_factory=lambda b, g: proc or FakeProc())
    eng._proc = proc or FakeProc()
    eng._sidecar = {"quant": "f16", "gguf_sha256": "abc",
                    "llamacpp_commit": "def",
                    "revision": "rev"}
    eng.model = lb.LlamaModelHandle(eng)
    eng.tokenizer = tok or FakeTok()
    eng.model_id = steer.MODEL_ID
    eng.revision = "rev"
    eng.rung = 1
    eng._reader = reader or FakeReader(steer.QWEN_SPEC)
    eng._stop_ids = [1]
    return eng


def _qwen_vec(layer=14, dose=1.0, scale=2.0):
    u = torch.zeros(steer.QWEN_SPEC.hidden)
    u[0] = 1.0
    return engine_mod.VectorEdit(layer, tuple(u.tolist()),
                                 scale, dose)


class CvecLayoutTest(unittest.TestCase):
    def test_empty_is_none(self):
        conv = lb.edits_to_cvec(engine_mod.EMPTY_EDITS, TINY,
                                {}, FakeReader())
        self.assertIsNone(conv)

    def test_slots_sum_and_range(self):
        reader = FakeReader()
        medians = {1: 2.0, 3: 4.0}
        edits = engine_mod.EditSet(
            neuron=(engine_mod.NeuronEdit(1, 2, 1.5),),
            vector=(engine_mod.VectorEdit(
                3, tuple([0.0] * 7 + [1.0]), 4.0, 0.5),))
        raw, lo, hi = lb.edits_to_cvec(edits, TINY, medians,
                                       reader)
        self.assertEqual((lo, hi), (1, 3))
        buf = np.frombuffer(raw, dtype=np.float32)
        self.assertEqual(buf.size, 4 * 8)
        w = reader.block
        cn = steer.column_norm_of(w, 2)
        want1 = (1.5 * 2.0 / cn) * w[:, 2].numpy()
        np.testing.assert_allclose(buf[0:8], want1, rtol=1e-6)
        np.testing.assert_allclose(buf[8:16], 0.0, atol=0.0)
        np.testing.assert_allclose(
            buf[16:24],
            np.array([0.0] * 7 + [4.0 * 0.5], dtype=np.float32),
            rtol=1e-6)
        self.assertEqual(buf[24:32].tolist(), [0.0] * 8)

    def test_neuron_and_vector_sum_on_one_layer(self):
        reader = FakeReader()
        edits = engine_mod.EditSet(
            neuron=(engine_mod.NeuronEdit(2, 0, 1.0),),
            vector=(engine_mod.VectorEdit(
                2, tuple([1.0] + [0.0] * 7), 3.0, 2.0),))
        raw, lo, hi = lb.edits_to_cvec(edits, TINY, {2: 5.0},
                                       reader)
        self.assertEqual((lo, hi), (2, 2))
        buf = np.frombuffer(raw, dtype=np.float32)
        w = reader.block
        want = ((1.0 * 5.0 / steer.column_norm_of(w, 0))
                * w[:, 0].numpy())
        want[0] += 3.0 * 2.0
        np.testing.assert_allclose(buf[8:16], want, rtol=1e-6)

    def test_ablation_rejects_whole_set(self):
        edits = engine_mod.EditSet(
            vector=(_qwen_vec(),),
            ablate=(engine_mod.Ablation(
                14, tuple([0.0] * steer.QWEN_SPEC.hidden)),))
        with self.assertRaises(lb.UnsupportedError):
            lb.edits_to_cvec(edits, steer.QWEN_SPEC, {},
                             FakeReader(steer.QWEN_SPEC))

    def test_read_rejects(self):
        edits = engine_mod.EditSet(
            read=(engine_mod.Read(14),))
        with self.assertRaises(lb.UnsupportedError):
            lb.edits_to_cvec(edits, steer.QWEN_SPEC, {},
                             FakeReader(steer.QWEN_SPEC))

    def test_layer_zero_rejects(self):
        reader = FakeReader()
        with self.assertRaises(lb.UnsupportedError):
            lb.edits_to_cvec(engine_mod.EditSet(
                neuron=(engine_mod.NeuronEdit(0, 1, 1.0),)),
                TINY, {0: 1.0}, reader)
        with self.assertRaises(lb.UnsupportedError):
            lb.edits_to_cvec(engine_mod.EditSet(
                vector=(engine_mod.VectorEdit(
                    0, tuple([0.0] * 8), 1.0, 1.0),)),
                TINY, {}, reader)

    def test_missing_median_names_torch(self):
        with self.assertRaises(lb.UnsupportedError) as ctx:
            lb.edits_to_cvec(engine_mod.EditSet(
                neuron=(engine_mod.NeuronEdit(1, 1, 1.0),)),
                TINY, {}, FakeReader())
        self.assertIn("LOBOT_BACKEND=torch", str(ctx.exception))

    def test_vector_shape_and_bounds(self):
        reader = FakeReader()
        with self.assertRaises(ValueError):
            lb.edits_to_cvec(engine_mod.EditSet(
                vector=(engine_mod.VectorEdit(
                    1, (1.0, 2.0), 1.0, 1.0),)),
                TINY, {}, reader)
        with self.assertRaises(ValueError):
            lb.edits_to_cvec(engine_mod.EditSet(
                vector=(engine_mod.VectorEdit(
                    9, tuple([0.0] * 8), 1.0, 1.0),)),
                TINY, {}, reader)


class ConversionMathTest(unittest.TestCase):
    """Torch-only proof: neuron pre-hook == F32 residual add."""

    def test_down_proj_linearity_is_exact(self):
        g = torch.Generator().manual_seed(7)
        lin = torch.nn.Linear(16, 8, bias=False)
        with torch.no_grad():
            lin.weight.copy_(torch.randn(8, 16, generator=g))
        x = torch.randn(2, 5, 16, generator=g)
        d, n = 1.7, 3
        xp = x.clone()
        xp[..., n] = xp[..., n] + d
        got = lin(xp)
        want = lin(x) + d * lin.weight[:, n]
        self.assertTrue(torch.allclose(got, want, atol=1e-5))

    def test_column_norm_shared(self):
        w = torch.randn(8, 16)
        self.assertAlmostEqual(
            steer.column_norm_of(w, 4),
            float(w.float()[:, 4].norm().item()))

    def test_column_norm_matches_model_path(self):
        lin = torch.nn.Linear(16, 8, bias=False)
        mlp = SimpleNamespace(down_proj=lin)
        block = SimpleNamespace(mlp=mlp)
        model = SimpleNamespace(
            model=SimpleNamespace(layers=[block]),
            config=SimpleNamespace(
                hidden_size=8, intermediate_size=16))
        spec = steer.ModelSpec("fake", 1, 8, 16, "llama", False)
        self.assertAlmostEqual(
            steer.column_norm(model, 0, 5, spec),
            steer.column_norm_of(lin.weight, 5))


# Engine.run keys: steer.generate_text keys plus hook_calls, reads,
# engine_seconds. LlamaEngine.run must carry the same set.
RUN_KEYS = {"text", "new_tokens", "token_ids", "seconds",
            "prefill_s", "decode_s", "tok_per_s", "peak_mib",
            "driver_mib", "cancelled", "hook_calls", "reads",
            "engine_seconds"}


class RunShapeTest(unittest.TestCase):
    def test_run_keys_match_engine(self):
        eng = _llama()
        got = eng.run("hi", engine_mod.EMPTY_EDITS, medians={},
                      max_new_tokens=8, seed=0)
        self.assertEqual(set(got.keys()), RUN_KEYS)
        self.assertEqual(got["token_ids"], [11, 12, 13])
        self.assertEqual(got["new_tokens"], 3)
        self.assertEqual(got["text"], "w11 w12 w13")
        self.assertFalse(got["cancelled"])
        self.assertIsNone(got["peak_mib"])
        self.assertIsNone(got["driver_mib"])
        self.assertEqual(got["reads"], {})

    def test_hook_calls_honest(self):
        eng = _llama()
        base = eng.run("hi", engine_mod.EMPTY_EDITS, medians={},
                       max_new_tokens=8, seed=0)
        self.assertEqual(base["hook_calls"], 0)
        one = eng.run(
            "hi", engine_mod.EditSet(vector=(_qwen_vec(),)),
            medians={}, max_new_tokens=8, seed=0)
        self.assertEqual(one["hook_calls"], 1 * (1 + 3))
        two = eng.run(
            "hi", engine_mod.EditSet(vector=(
                _qwen_vec(layer=14), _qwen_vec(layer=13),)),
            medians={}, max_new_tokens=8, seed=0)
        self.assertEqual(two["hook_calls"], 2 * (1 + 3))

    def test_run_sends_cvec_and_stop(self):
        proc = FakeProc()
        eng = _llama(proc=proc)
        eng.run("hi", engine_mod.EditSet(vector=(_qwen_vec(),)),
                medians={}, max_new_tokens=8, seed=0)
        sent = proc.generates[0]
        self.assertTrue(sent["cvec"])
        self.assertEqual(sent["il"], (14, 14))
        self.assertEqual(sent["stop_ids"], [1])
        self.assertEqual(sent["ids"], [5, 6, 7])

    def test_run_empty_clears_cvec(self):
        proc = FakeProc()
        eng = _llama(proc=proc)
        eng.run("hi", engine_mod.EMPTY_EDITS, medians={},
                max_new_tokens=8, seed=0)
        self.assertFalse(proc.generates[0]["cvec"])

    def test_run_cancelled_before_start(self):
        proc = FakeProc()
        eng = _llama(proc=proc)
        cancel = threading.Event()
        cancel.set()
        got = eng.run("hi", engine_mod.EditSet(
            vector=(_qwen_vec(),)), medians={},
            max_new_tokens=8, seed=0, cancel=cancel)
        self.assertTrue(got["cancelled"])
        self.assertEqual(got["token_ids"], [])
        self.assertEqual(proc.generates, [])

    def test_run_daemon_error_raises(self):
        class Dead(FakeProc):
            def generate(self, *a, **k):
                yield ("done", {"stop": "error",
                                "error": "boom",
                                "cancelled": False,
                                "prefill_ms": 0.0,
                                "decode_ms": 0.0})

        eng = _llama(proc=Dead())
        with self.assertRaises(lb.DaemonError):
            eng.run("hi", engine_mod.EMPTY_EDITS, medians={},
                    max_new_tokens=8, seed=0)

    def test_stream_events_and_done(self):
        eng = _llama()
        evs = list(eng.run_stream(
            "hi", engine_mod.EMPTY_EDITS, medians={},
            max_new_tokens=8, seed=0))
        toks = [e["text"] for e in evs
                if e["type"] == "token"]
        self.assertEqual("".join(toks), "w11 w12 w13")
        self.assertEqual(evs[-1]["type"], "done")
        self.assertEqual(
            set(evs[-1]["result"].keys()), RUN_KEYS)

    def test_stream_dial_raises(self):
        eng = _llama()
        with self.assertRaises(lb.UnsupportedError):
            list(eng.run_stream(
                "hi", engine_mod.EMPTY_EDITS, medians={},
                max_new_tokens=8, seed=0,
                dial={"trait": "x"}))

    def test_stream_close_cancels_proc(self):
        proc = FakeProc()
        eng = _llama(proc=proc)
        stream = eng.run_stream(
            "hi", engine_mod.EMPTY_EDITS, medians={},
            max_new_tokens=8, seed=0)
        first = next(stream)
        self.assertEqual(first["type"], "token")
        stream.close()
        self.assertTrue(proc.stream_closed)

    def test_batched_shape(self):
        eng = _llama()
        out = eng.run_batched(
            "hi", [engine_mod.EMPTY_EDITS,
                   engine_mod.EditSet(vector=(_qwen_vec(),))],
            medians={}, max_new_tokens=8, seed=0)
        self.assertFalse(out["fallback_used"])
        self.assertEqual(len(out["rows"]), 2)
        self.assertEqual(
            out["hook_calls"],
            sum(r["hook_calls"] for r in out["rows"]))
        self.assertEqual(out["reads"], {})

    def test_batched_stream_columns(self):
        eng = _llama()
        evs = list(eng.run_batched_stream(
            "hi", [engine_mod.EMPTY_EDITS,
                   engine_mod.EMPTY_EDITS],
            medians={}, max_new_tokens=8, seed=0))
        cols = sorted({e["col"] for e in evs
                       if e["type"] == "token"})
        self.assertEqual(cols, [0, 1])
        done = evs[-1]
        self.assertEqual(done["type"], "done")
        self.assertFalse(done["result"]["fallback_used"])
        self.assertEqual(len(done["result"]["rows"]), 2)

    def test_batched_empty_raises(self):
        eng = _llama()
        with self.assertRaises(ValueError):
            eng.run_batched("hi", [], medians={},
                            max_new_tokens=8, seed=0)


class PrefillTest(unittest.TestCase):
    def test_prefill_shape_and_top(self):
        proc = FakeProc(vocab=64, top_logit=7)
        eng = _llama(proc=proc)
        vec = eng.prefill("some question")
        self.assertEqual(tuple(vec.shape), (64,))
        self.assertEqual(vec.dtype, torch.float32)
        self.assertEqual(int(vec.argmax()), 7)
        ids, cvec = proc.prefills[0]
        self.assertEqual(ids, [5, 6, 7])
        self.assertIsNone(cvec)

    def test_prefill_with_edits_sends_cvec(self):
        proc = FakeProc()
        eng = _llama(proc=proc)
        eng.prefill("q", engine_mod.EditSet(
            vector=(_qwen_vec(),)), medians={})
        _ids, cvec = proc.prefills[0]
        self.assertIsNotNone(cvec)
        b64, lo, hi = cvec
        self.assertEqual((lo, hi), (14, 14))
        raw = base64.b64decode(b64)
        self.assertEqual(len(raw),
                         32 * 2560 * 4)

    def test_prefill_edits_need_medians(self):
        eng = _llama()
        with self.assertRaises(ValueError):
            eng.prefill("q", engine_mod.EditSet(
                vector=(_qwen_vec(),)))

    def test_prefill_vocab_mismatch_raises(self):
        class Short(FakeProc):
            def prefill(self, ids, cvec):
                return b"\x00" * 16

        eng = _llama(proc=Short())
        with self.assertRaises(lb.DaemonError):
            eng.prefill("q")


class HandleTest(unittest.TestCase):
    def test_call_matches_judge_pattern(self):
        proc = FakeProc(vocab=64, top_logit=9)
        eng = _llama(proc=proc)
        ids = torch.tensor([[5, 6, 7]])
        logits = eng.model(ids).logits[0, -1].float()
        self.assertEqual(tuple(logits.shape), (64,))
        self.assertEqual(int(logits.argmax()), 9)
        sent, cvec = proc.prefills[0]
        self.assertEqual(sent, [5, 6, 7])
        self.assertIsNone(cvec)

    def test_call_rejects_shape(self):
        eng = _llama()
        with self.assertRaises(lb.UnsupportedError):
            eng.model(torch.tensor([1, 2, 3]))
        with self.assertRaises(lb.UnsupportedError):
            eng.model(torch.zeros(2, 3, dtype=torch.long))

    def test_logits_index_restricted(self):
        proc = FakeProc()
        eng = _llama(proc=proc)
        out = eng.model(torch.tensor([[5]]))
        with self.assertRaises(lb.UnsupportedError):
            _ = out.logits[0]
        with self.assertRaises(lb.UnsupportedError):
            _ = out.logits[0, 0]

    def test_attr_raises_torch_only(self):
        eng = _llama()
        with self.assertRaises(lb.UnsupportedError) as ctx:
            _ = eng.model.parameters
        self.assertIn("LOBOT_BACKEND=torch", str(ctx.exception))
        with self.assertRaises(lb.UnsupportedError):
            _ = eng.model.config


class BackendSwitchTest(unittest.TestCase):
    def test_engine_supports_all(self):
        eng = engine_mod.Engine()
        self.assertEqual(eng.backend, "torch")
        for f in engine_mod.FEATURES:
            self.assertTrue(eng.supports(f))

    def test_llama_supports_none(self):
        eng = lb.LlamaEngine()
        self.assertEqual(eng.backend, "llama")
        for f in engine_mod.FEATURES:
            self.assertFalse(eng.supports(f))

    def test_capture_raises(self):
        eng = _llama()
        with self.assertRaises(lb.UnsupportedError):
            eng.capture(["hi"], [1])
        with self.assertRaises(lb.UnsupportedError):
            eng.capture_with_edits(
                ["hi"], [1], engine_mod.EMPTY_EDITS, {})

    def test_fingerprint_keys(self):
        eng = _llama()
        fp = eng.fingerprint()
        for k in ("model_id", "revision", "backend",
                  "gguf_sha256", "quant", "llamacpp_commit"):
            self.assertIn(k, fp)
        self.assertEqual(fp["backend"], "llama")
        self.assertEqual(fp["quant"], "f16")

    def test_stop_ids_prefer_text_config(self):
        from unittest import mock
        with mock.patch("transformers.AutoConfig") as ac:
            ac.from_pretrained.return_value \
                .get_text_config.return_value = SimpleNamespace(
                    eos_token_id=[248044])
            self.assertEqual(
                lb.LlamaEngine._resolve_stop_ids("m", FakeTok()),
                [248044])
            ac.from_pretrained.side_effect = RuntimeError("no")
            self.assertEqual(
                lb.LlamaEngine._resolve_stop_ids("m", FakeTok()),
                [1])

    def test_get_engine_switch(self):
        import os
        saved_engine = engine_mod._ENGINE
        saved_env = os.environ.get("LOBOT_BACKEND")
        try:
            engine_mod._ENGINE = None
            os.environ.pop("LOBOT_BACKEND", None)
            self.assertIsInstance(engine_mod.get_engine(),
                                  engine_mod.Engine)
            engine_mod._ENGINE = None
            os.environ["LOBOT_BACKEND"] = "llama"
            self.assertIsInstance(engine_mod.get_engine(),
                                  lb.LlamaEngine)
            engine_mod._ENGINE = None
            os.environ["LOBOT_BACKEND"] = "bogus"
            with self.assertRaises(ValueError):
                engine_mod.get_engine()
        finally:
            if saved_env is None:
                os.environ.pop("LOBOT_BACKEND", None)
            else:
                os.environ["LOBOT_BACKEND"] = saved_env
            engine_mod._ENGINE = saved_engine

    def test_load_dim_checks(self):
        from unittest import mock
        sidecar = {"quant": "f16", "gguf_sha256": "s",
                   "llamacpp_commit": "c",
                   "revision": "rev" * 10}
        with mock.patch.object(
                lb, "resolve_gguf",
                return_value=(Path(__file__), sidecar)), \
                mock.patch.object(
                    steer, "load_tokenizer",
                    return_value=FakeTok(vocab=64)), \
                mock.patch.object(
                    lb, "DownProjReader",
                    return_value=FakeReader(steer.QWEN_SPEC)), \
                mock.patch.object(
                    lb.LlamaEngine, "_resolve_stop_ids",
                    return_value=[1]), \
                mock.patch.object(
                    lb, "torch_vocab_size",
                    return_value=64):
            bad = FakeProc(vocab=64, layers=3, hidden=8)
            eng = lb.LlamaEngine(
                proc_factory=lambda b, g: bad)
            with self.assertRaises(RuntimeError):
                eng.load(steer.MODEL_ID, revision="rev" * 10)
            self.assertTrue(bad.closed)
            bad2 = FakeProc(vocab=65, layers=32, hidden=2560)
            eng2 = lb.LlamaEngine(
                proc_factory=lambda b, g: bad2)
            with self.assertRaises(RuntimeError):
                eng2.load(steer.MODEL_ID, revision="rev" * 10)
            self.assertTrue(bad2.closed)
            good = FakeProc(vocab=64, layers=32, hidden=2560)
            eng3 = lb.LlamaEngine(
                proc_factory=lambda b, g: good)
            self.assertEqual(
                eng3.load(steer.MODEL_ID, revision="rev" * 10),
                1)
            self.assertTrue(eng3.warm)
            self.assertIsNotNone(eng3.model)
            eng3.unload()
            self.assertFalse(eng3.warm)
            self.assertTrue(good.closed)


if __name__ == "__main__":
    unittest.main()
