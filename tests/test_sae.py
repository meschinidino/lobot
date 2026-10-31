"""Unit tests: clean features SAE (F1).

Tiny fake llama-layout models (2 layers, hidden 8, intermediate 16,
vocab 16) plus random activations. No weights. Proves: the TopK SAE
keeps exact sparsity and reconstructs; training lowers MSE and keeps
decoder rows unit-norm; save/load round-trips; dead counts mark
silent features; the activation cache and feature fires round-trip;
pushes_of_direction matches the hand projection; steer_lift_vector
is 1.0 at zero dose and above 1.0 for a planted direction; the
feature routes list, card and lift against a tiny trained SAE.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import engine  # noqa: E402
from model_lobotomy import reader  # noqa: E402
from model_lobotomy import sae  # noqa: E402
from model_lobotomy import server  # noqa: E402
from model_lobotomy import steer  # noqa: E402

VOCAB = ["<bos>", "<unk>", "pirate", "ship", "sea", "shiver", "timbers",
         "yo", "ho", "bottle", "rum", "map", "gold", "storm", "captain",
         "parrot"]
LAYERS, HIDDEN, INTER, FEATS, TOPK = 2, 8, 16, 32, 4


def make_spec():
    return steer.ModelSpec("fake-sae", layers=LAYERS, hidden=HIDDEN,
                           intermediate=INTER, block_path="llama",
                           thinking_kw=False)


class FakeNorm(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden))

    def forward(self, x):
        return x * self.weight


class FakeMLP(torch.nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)


class FakeBlock(torch.nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.mlp = FakeMLP(hidden, intermediate)

    def forward(self, x):
        return x + self.mlp.down_proj(self.mlp.up_proj(x))


class FakeInner(torch.nn.Module):
    def __init__(self, layers, hidden, intermediate):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [FakeBlock(hidden, intermediate) for _ in range(layers)])
        self.norm = FakeNorm(hidden)


class FakeModel(torch.nn.Module):
    def __init__(self, layers=LAYERS, hidden=HIDDEN,
                 intermediate=INTER, vocab=len(VOCAB)):
        super().__init__()
        self.model = FakeInner(layers, hidden, intermediate)
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(
            num_hidden_layers=layers, hidden_size=hidden,
            intermediate_size=intermediate)

    def forward(self, ids):
        x = self.embed(ids)
        for block in self.model.layers:
            x = block(x)
        x = self.model.norm(x)
        return SimpleNamespace(logits=self.lm_head(x))


class FakeTok:
    chat_template = "fake"

    def __init__(self, vocab=None):
        self.vocab = list(vocab or VOCAB)
        self.inv = {w: i for i, w in enumerate(self.vocab)}

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        return "yo " + messages[0]["content"]

    def __call__(self, text, return_tensors=None,
                 add_special_tokens=True, **kw):
        ids = [self.inv.get(w.lower(), 1) for w in text.split()]
        if add_special_tokens:
            ids = [0] + ids
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": [ids]}

    def decode(self, ids, skip_special_tokens=False):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return "".join(self.vocab[int(i)] for i in ids)


def make_eng(seed=0):
    torch.manual_seed(seed)
    eng = engine.Engine()
    eng.model = FakeModel()
    eng.tokenizer = FakeTok()
    eng.model_id = "fake-sae"
    eng.revision = "test-rev"
    return eng


class SaeCase(unittest.TestCase):
    def setUp(self):
        steer.MODEL_SPECS["fake-sae"] = make_spec()
        self.addCleanup(steer.MODEL_SPECS.pop, "fake-sae", None)
        self.engs = []

    def tearDown(self):
        for eng in self.engs:
            eng.remove_all()

    def make_eng(self, seed=0):
        eng = make_eng(seed)
        self.engs.append(eng)
        return eng

    def tmpdir(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def assert_clean(self, eng):
        self.assertEqual(eng.hook_count(), 0)
        self.assertEqual(steer.steering_hook_count(), 0)
        for block in eng.model.model.layers:
            self.assertEqual(
                len(block.mlp.down_proj._forward_pre_hooks), 0)
            self.assertEqual(len(block._forward_hooks), 0)
            self.assertEqual(len(block._forward_pre_hooks), 0)


class TopkTest(SaeCase):
    def test_topk_forward_shapes_and_sparsity(self):
        torch.manual_seed(0)
        model = sae.TopkSAE(HIDDEN, FEATS, TOPK, seed=0)
        x = torch.randn(10, HIDDEN)
        recon, codes = model(x)
        self.assertEqual(list(recon.shape), [10, HIDDEN])
        self.assertEqual(list(codes.shape), [10, FEATS])
        self.assertTrue(bool(((codes != 0).sum(dim=1) == TOPK).all()))

    def test_bad_shapes_rejected(self):
        with self.assertRaises(ValueError):
            sae.TopkSAE(0, FEATS, TOPK)
        with self.assertRaises(ValueError):
            sae.TopkSAE(HIDDEN, FEATS, 0)
        with self.assertRaises(ValueError):
            sae.TopkSAE(HIDDEN, FEATS, FEATS + 1)


class TrainTest(SaeCase):
    def test_train_loss_falls(self):
        torch.manual_seed(1)
        acts = torch.randn(256, HIDDEN)
        _, stats = sae.train_sae(acts, hidden=HIDDEN, features=FEATS,
                                 top_k=TOPK, epochs=2, batch=64,
                                 seed=0)
        self.assertLess(stats["mse_last"], stats["mse_first"])
        self.assertEqual(stats["live"] + stats["dead"], FEATS)
        self.assertEqual(stats["steps"], 8)

    def test_decoder_rows_unit_norm_after_train(self):
        torch.manual_seed(2)
        acts = torch.randn(128, HIDDEN)
        model, _ = sae.train_sae(acts, hidden=HIDDEN, features=FEATS,
                                 top_k=TOPK, epochs=1, batch=64,
                                 seed=0)
        norms = model.W_dec.detach().norm(dim=1)
        for n in norms.tolist():
            self.assertAlmostEqual(n, 1.0, places=5)

    def test_save_load_round_trip(self):
        torch.manual_seed(3)
        acts = torch.randn(128, HIDDEN)
        model, stats = sae.train_sae(acts, hidden=HIDDEN,
                                     features=FEATS, top_k=TOPK,
                                     epochs=1, batch=64, seed=0)
        out = self.tmpdir() / "sae"
        sae.save_sae(out, model, {"layer": 1, **stats})
        back, cfg = sae.load_sae(out)
        self.assertEqual((cfg["hidden"], cfg["features"], cfg["top_k"]),
                         (HIDDEN, FEATS, TOPK))
        x = torch.randn(7, HIDDEN)
        with torch.no_grad():
            a = model.encode(x)
            b = back.encode(x)
        self.assertTrue(bool(torch.equal(a, b)))

    def test_dead_count_marks_silent_features(self):
        torch.manual_seed(4)
        acts = torch.zeros(64, HIDDEN)
        acts[:, 0] = 5.0
        _, stats = sae.train_sae(acts, hidden=HIDDEN, features=FEATS,
                                 top_k=1, epochs=1, batch=64, seed=0)
        self.assertGreater(stats["dead"], 0)
        self.assertEqual(stats["live"] + stats["dead"], FEATS)

    def test_feature_direction_unit_and_bounds(self):
        model = sae.TopkSAE(HIDDEN, FEATS, TOPK, seed=0)
        vec = sae.feature_direction(model, 3)
        self.assertAlmostEqual(float(vec.norm().item()), 1.0, places=5)
        with self.assertRaises(ValueError):
            sae.feature_direction(model, FEATS)


class CacheTest(SaeCase):
    def test_cache_round_trip(self):
        eng = self.make_eng(seed=1)
        docs = ["pirate ship sea", "storm map gold", "rum bottle yo"]
        got = sae.cache_block_acts(eng, docs, 1)
        # 3 docs x (1 BOS + 3 words), BOS skipped -> 9 rows.
        self.assertEqual(list(got["acts"].shape), [9, HIDDEN])
        self.assertEqual(len(got["tokens"]), 12)
        self.assertEqual(got["doc_starts"], [0, 4, 8])
        self.assertEqual(got["tokens"][1], "pirate")
        self.assert_clean(eng)

    def test_cache_rejects_bad_input(self):
        eng = self.make_eng(seed=1)
        with self.assertRaises(ValueError):
            sae.cache_block_acts(eng, [], 0)
        with self.assertRaises(ValueError):
            sae.cache_block_acts(eng, ["sea"], 99)
        self.assert_clean(eng)

    def test_fires_round_trip(self):
        eng = self.make_eng(seed=2)
        docs = ["pirate ship sea", "storm map gold", "rum bottle yo"]
        cached = sae.cache_block_acts(eng, docs, 0)
        torch.manual_seed(0)
        model = sae.TopkSAE(HIDDEN, FEATS, TOPK, seed=0)
        out = self.tmpdir() / "sae"
        sae.build_feature_fires(model, cached["acts"], cached["tokens"],
                                cached["doc_starts"], out, k=3)
        entries = sae.feature_fires(out, 5, k=3)
        self.assertLessEqual(len(entries), 3)
        acts_desc = [e["activation"] for e in entries]
        self.assertTrue(all(a >= b for a, b in
                            zip(acts_desc, acts_desc[1:])))
        for e in entries:
            start, end = e["mark"]
            self.assertEqual(e["context"][start:end], e["token"])
        with self.assertRaises(ValueError):
            sae.feature_fires(out, FEATS)
        self.assert_clean(eng)


class DirectionTest(SaeCase):
    def test_pushes_of_direction_matches_manual_projection(self):
        eng = self.make_eng(seed=1)
        vec = torch.randn(HIDDEN)
        got = reader.pushes_of_direction(eng, vec, k=5, layer=1)
        gamma = eng.model.model.norm.weight.detach().float()
        head = eng.model.lm_head.weight.detach().float()
        logits = head @ (vec.float() * gamma)
        up_vals, up_idx = torch.topk(logits, 5, largest=True, sorted=True)
        self.assertEqual([e["id"] for e in got["up"]], up_idx.tolist())
        for entry, val in zip(got["up"], up_vals.tolist()):
            self.assertAlmostEqual(entry["logit"], val, places=5)
        self.assertAlmostEqual(got["layer_frac"], 1.0)
        none_layer = reader.pushes_of_direction(eng, vec, k=5)
        self.assertIsNone(none_layer["layer_frac"])
        self.assert_clean(eng)

    def test_pushes_of_direction_matches_neuron_column(self):
        eng = self.make_eng(seed=2)
        card = reader.pushes_toward(eng, 1, 3, k=6)
        down = steer.resolve_down_proj(
            eng.model, 1, eng.spec).weight.detach().float()
        vec = down[:, 3]
        got = reader.pushes_of_direction(eng, vec, k=6, layer=1)
        self.assertEqual(got, card)
        self.assert_clean(eng)

    def test_pushes_of_direction_rejects_bad_width(self):
        eng = self.make_eng(seed=1)
        with self.assertRaises(ValueError):
            reader.pushes_of_direction(eng, [1.0, 2.0])
        with self.assertRaises(ValueError):
            reader.pushes_of_direction(eng, [float("nan")] * HIDDEN)
        self.assert_clean(eng)

    def _plant(self, eng):
        model = eng.model
        star = VOCAB.index("rum")
        dim = 1
        with torch.no_grad():
            head = model.lm_head.weight
            head.zero_()
            rnd = torch.randn(len(VOCAB), HIDDEN) * 0.05
            rnd[:, dim] = 0.0
            head[:, :] = rnd
            head[star] = torch.tensor([0., 1.5, 0, 0, 0, 0, 0, 0])
            model.model.layers[1].mlp.up_proj.weight.zero_()

    def test_lift_vector_above_one_for_planted_direction(self):
        eng = self.make_eng(seed=5)
        self._plant(eng)
        direction = [0.0] * HIDDEN
        direction[1] = 2.0  # non-unit on purpose: normalized inside
        got = reader.steer_lift_vector(
            eng, 1, direction, 2.0, ["rum"],
            ["sea storm", "ship map"], {1: 1.0})
        self.assertEqual(got["words"], ["rum"])
        self.assertGreater(got["steered_p"][0], got["base_p"][0])
        self.assertGreater(got["lift"], 1.0)
        self.assert_clean(eng)

    def test_lift_vector_equals_one_at_zero_dose(self):
        eng = self.make_eng(seed=5)
        self._plant(eng)
        direction = [0.0] * HIDDEN
        direction[1] = 1.0
        got = reader.steer_lift_vector(
            eng, 1, direction, 0.0, ["rum"],
            ["sea storm", "ship map"], {1: 1.0})
        self.assertAlmostEqual(got["lift"], 1.0)
        self.assertAlmostEqual(got["base_p"][0], got["steered_p"][0])
        with self.assertRaises(ValueError):
            reader.steer_lift_vector(
                eng, 1, direction, 1.0, [], ["sea"], {1: 1.0})
        with self.assertRaises(ValueError):
            reader.steer_lift_vector(
                eng, 1, [0.0] * HIDDEN, 1.0, ["rum"], ["sea"],
                {1: 1.0})
        with self.assertRaises(ValueError):
            reader.steer_lift_vector(
                eng, 1, direction, 1.0, ["rum"], ["sea"], {})
        self.assert_clean(eng)


class TopWordsTest(SaeCase):
    def test_top_words_picks_whole_words(self):
        pushes = {"up": [{"token": " x"}, {"token": "pirate"},
                          {"token": "ship"}, {"token": "sea"},
                          {"token": "sh"}]}
        self.assertEqual(sae.top_words(pushes),
                         ["pirate", "ship", "sea"])

    def test_top_words_falls_back_to_top_token(self):
        pushes = {"up": [{"token": " x"}, {"token": "##s"}]}
        self.assertEqual(sae.top_words(pushes), [" x"])


class FeatureCardTest(SaeCase):
    def test_card_shape_with_cache(self):
        eng = self.make_eng(seed=2)
        docs = ["pirate ship sea", "storm map gold", "rum bottle yo"]
        cached = sae.cache_block_acts(eng, docs, 1)
        torch.manual_seed(0)
        model = sae.TopkSAE(HIDDEN, FEATS, TOPK, seed=0)
        out = self.tmpdir() / "sae"
        sae.build_feature_fires(model, cached["acts"], cached["tokens"],
                                cached["doc_starts"], out, k=4)
        card = sae.feature_card(eng, model, 1, 5, cache_dir=out, k=4)
        self.assertEqual(
            (card["model_id"], card["layer"], card["index"]),
            ("fake-sae", 1, 5))
        self.assertIn("coherence", card["pushes_toward"])
        self.assertEqual(len(card["pushes_toward"]["up"]), 4)
        self.assertEqual(card["fires_on_cache"],
                         {"built": True, "tokens": 12})
        self.assertTrue(card["live"])
        self.assert_clean(eng)

    def test_card_without_cache_has_null_fires(self):
        eng = self.make_eng(seed=2)
        model = sae.TopkSAE(HIDDEN, FEATS, TOPK, seed=0)
        card = sae.feature_card(eng, model, 0, 0, cache_dir=None, k=4)
        self.assertIsNone(card["fires_on"])
        self.assertEqual(card["fires_on_cache"],
                         {"built": False, "tokens": 0})
        self.assert_clean(eng)


class FakeRouteEngine:
    """Warm stub for feature routes: no model math, real lock."""

    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.model = object()
        self.model_id = "Qwen/Qwen3.5-4B"
        self.revision = "stub-rev"
        self.tokenizer = object()

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


class FeatureRoutesTest(unittest.TestCase):
    def setUp(self):
        self.eng = FakeRouteEngine()
        self.patches = [
            mock.patch.object(server, "NO_MODEL", False),
            mock.patch.object(server, "get_engine",
                              return_value=self.eng),
            mock.patch.object(steer, "calibration_cache_key",
                              return_value={}),
            mock.patch.object(
                steer, "calibrate_layer_medians",
                return_value=({}, {"cache_hit": True,
                                   "key": {"prompts_sha256": "abc"}})),
        ]
        for p in self.patches:
            p.start()
        self.addCleanup(server._sae_cache.clear)
        self.client = TestClient(server.api)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def _tmp_root(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def _write_sae(self, root, layer=5):
        from model_lobotomy import app as engine_mod
        slug = server._model_slug(engine_mod._active_model_id)
        out = root / "store" / "sae" / slug / f"L{layer}"
        torch.manual_seed(0)
        model = sae.TopkSAE(HIDDEN, FEATS, TOPK, seed=0)
        sae.save_sae(out, model, {"model_id": engine_mod._active_model_id,
                                  "layer": layer, "live": 30, "dead": 2,
                                  "mse_last": 0.5})
        return out

    def _stub_reader(self):
        stub = mock.Mock()
        stub.pushes_of_direction.return_value = {
            "up": [{"token": t, "id": i, "logit": 3.0 - i}
                   for i, t in enumerate(["pirate", "ship", "sea"])],
            "down": [], "coherence": 1.0, "layer_frac": 0.2}
        stub.steer_lift_vector.return_value = {
            "words": ["pirate"], "base_p": [0.1], "steered_p": [0.4],
            "lift": 4.0}
        return stub

    def test_features_empty_without_saes(self):
        root = self._tmp_root()
        with mock.patch.object(server, "ROOT", root):
            r = self.client.get("/api/features")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), {"saes": []})

    def test_features_lists_trained_sae(self):
        from model_lobotomy import app as engine_mod
        root = self._tmp_root()
        self._write_sae(root, layer=5)
        with mock.patch.object(server, "ROOT", root):
            r = self.client.get("/api/features")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), {"saes": [{
            "model_id": engine_mod._active_model_id, "layer": 5,
            "features": FEATS, "top_k": TOPK, "live": 30, "dead": 2,
            "mse": 0.5}]})

    def test_feature_card_shape(self):
        from model_lobotomy import app as engine_mod
        root = self._tmp_root()
        out = self._write_sae(root, layer=5)
        torch.manual_seed(0)
        model = sae.TopkSAE(HIDDEN, FEATS, TOPK, seed=0)
        acts = torch.randn(24, HIDDEN)
        tokens = ["w%d" % i for i in range(24)]
        sae.build_feature_fires(model, acts, tokens, [0, 12], out, k=3)
        stub = self._stub_reader()
        with mock.patch.object(server, "ROOT", root), \
                mock.patch.object(server, "reader", stub), \
                mock.patch.object(sae, "reader", stub):
            r = self.client.get("/api/feature/5/7?k=3")
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d["model_id"], engine_mod._active_model_id)
        self.assertEqual((d["layer"], d["index"]), (5, 7))
        self.assertEqual(d["pushes_toward"]["coherence"], 1.0)
        self.assertEqual(d["fires_on_cache"],
                         {"built": True, "tokens": 24})
        stub.pushes_of_direction.assert_called_once()
        self.assertFalse(self.eng.lock.locked())

    def test_feature_card_404_without_sae(self):
        root = self._tmp_root()
        with mock.patch.object(server, "ROOT", root):
            r = self.client.get("/api/feature/5/7")
        self.assertEqual(r.status_code, 404)

    def test_feature_card_400_bad_index(self):
        root = self._tmp_root()
        self._write_sae(root, layer=5)
        stub = self._stub_reader()
        with mock.patch.object(server, "ROOT", root), \
                mock.patch.object(server, "reader", stub):
            r = self.client.get(f"/api/feature/5/{FEATS}")
        self.assertEqual(r.status_code, 400)

    def test_feature_lift_shape(self):
        root = self._tmp_root()
        self._write_sae(root, layer=5)
        stub = self._stub_reader()
        with mock.patch.object(server, "ROOT", root), \
                mock.patch.object(server, "reader", stub):
            r = self.client.post("/api/feature-lift", json={
                "layer": 5, "index": 7, "dose": 1.0,
                "words": ["pirate"], "prompts": ["Say hi."]})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["lift"], 4.0)
        _, args, _ = stub.steer_lift_vector.mock_calls[0]
        self.assertEqual(tuple(args[1:4]), (5, args[2], 1.0))
        self.assertEqual(args[4], ["pirate"])
        self.assertFalse(self.eng.lock.locked())

    def test_feature_lift_default_words_from_pushes(self):
        root = self._tmp_root()
        self._write_sae(root, layer=5)
        stub = self._stub_reader()
        with mock.patch.object(server, "ROOT", root), \
                mock.patch.object(server, "reader", stub):
            r = self.client.post("/api/feature-lift", json={
                "layer": 5, "index": 7, "dose": 1.0,
                "prompts": ["Say hi."]})
        self.assertEqual(r.status_code, 200, r.text)
        _, args, _ = stub.steer_lift_vector.mock_calls[0]
        self.assertEqual(args[4], ["pirate", "ship", "sea"])

    def test_feature_lift_404_without_sae(self):
        root = self._tmp_root()
        with mock.patch.object(server, "ROOT", root):
            r = self.client.post("/api/feature-lift", json={
                "layer": 5, "index": 7, "dose": 1.0,
                "words": ["pirate"]})
        self.assertEqual(r.status_code, 404)

    def test_feature_routes_503_without_weights(self):
        root = self._tmp_root()
        self._write_sae(root, layer=5)
        with mock.patch.object(server, "ROOT", root), \
                mock.patch.object(server, "NO_MODEL", True):
            r = self.client.get("/api/feature/5/7")
            self.assertEqual(r.status_code, 503)
            r = self.client.post("/api/feature-lift", json={
                "layer": 5, "index": 7, "dose": 1.0,
                "words": ["pirate"], "prompts": ["Say hi."]})
            self.assertEqual(r.status_code, 503)

    def test_feature_routes_409_when_busy(self):
        root = self._tmp_root()
        self._write_sae(root, layer=5)
        self.eng.lock.acquire()
        try:
            with mock.patch.object(server, "ROOT", root):
                r = self.client.get("/api/feature/5/7")
                self.assertEqual(r.status_code, 409)
                r = self.client.post("/api/feature-lift", json={
                    "layer": 5, "index": 7, "dose": 1.0,
                    "words": ["pirate"], "prompts": ["Say hi."]})
                self.assertEqual(r.status_code, 409)
        finally:
            self.eng.lock.release()

    def test_feature_routes_reject_cross_origin_post(self):
        root = self._tmp_root()
        self._write_sae(root, layer=5)
        with mock.patch.object(server, "ROOT", root):
            r = self.client.post(
                "/api/feature-lift",
                json={"layer": 5, "index": 7, "dose": 1.0,
                      "words": ["pirate"]},
                headers={"Origin": "http://evil.test"})
        self.assertEqual(r.status_code, 403)


if __name__ == "__main__":
    unittest.main()
