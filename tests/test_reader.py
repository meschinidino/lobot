"""Unit tests: reader core library (R1).

Tiny fake llama-layout models (2 layers, hidden 4, intermediate 8,
vocab 16) with a real nn.Linear lm_head and a scaling final norm.
No weights. Proves: pushes_toward equals the hand projection;
coherence grades whole words; the fires_on cache round-trips and
skips BOS; find/word-neurons/lift each surface a planted neuron;
every call leaves zero hooks, also on failure.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import engine  # noqa: E402
from model_lobotomy import reader  # noqa: E402
from model_lobotomy import steer  # noqa: E402

VOCAB = ["<bos>", "<unk>", "pirate", "ship", "sea", "shiver", "timbers",
         "yo", "ho", "bottle", "rum", "map", "gold", "storm", "captain",
         "parrot"]
LAYERS, HIDDEN, INTER = 2, 4, 8


def make_spec():
    return steer.ModelSpec("fake-reader", layers=LAYERS, hidden=HIDDEN,
                           intermediate=INTER, block_path="llama",
                           thinking_kw=False)


class FakeNorm(torch.nn.Module):
    """Scaling norm with a real .weight (no re-normalization)."""

    def __init__(self, hidden):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden))

    def forward(self, x):
        return x * self.weight


class FakeReaderMLP(torch.nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)


class FakeReaderBlock(torch.nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.mlp = FakeReaderMLP(hidden, intermediate)

    def forward(self, x):
        return x + self.mlp.down_proj(self.mlp.up_proj(x))


class FakeInner(torch.nn.Module):
    def __init__(self, layers, hidden, intermediate):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [FakeReaderBlock(hidden, intermediate) for _ in range(layers)])
        self.norm = FakeNorm(hidden)


class FakeReaderModel(torch.nn.Module):
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
    """Word tokenizer: BOS + per-word ids, unknown words map to <unk>."""

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
    eng.model = FakeReaderModel()
    eng.tokenizer = FakeTok()
    eng.model_id = "fake-reader"
    eng.revision = "test-rev"
    return eng


class ReaderCase(unittest.TestCase):
    def setUp(self):
        steer.MODEL_SPECS["fake-reader"] = make_spec()
        self.addCleanup(steer.MODEL_SPECS.pop, "fake-reader", None)
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


class PushesTowardTest(ReaderCase):
    def test_pushes_toward_matches_manual_projection(self):
        eng = self.make_eng(seed=1)
        got = reader.pushes_toward(eng, 1, 3, k=5)
        down = steer.resolve_down_proj(
            eng.model, 1, eng.spec).weight.detach().float()
        gamma = eng.model.model.norm.weight.detach().float()
        head = eng.model.lm_head.weight.detach().float()
        logits = head @ (down[:, 3] * gamma)
        up_vals, up_idx = torch.topk(logits, 5, largest=True, sorted=True)
        down_vals, down_idx = torch.topk(
            logits, 5, largest=False, sorted=True)
        self.assertEqual([e["id"] for e in got["up"]], up_idx.tolist())
        self.assertEqual([e["id"] for e in got["down"]], down_idx.tolist())
        for entry, val in zip(got["up"], up_vals.tolist()):
            self.assertAlmostEqual(entry["logit"], val, places=5)
            self.assertEqual(entry["token"], VOCAB[entry["id"]])
        for entry, val in zip(got["down"], down_vals.tolist()):
            self.assertAlmostEqual(entry["logit"], val, places=5)
        self.assertAlmostEqual(
            got["coherence"],
            reader.coherence([e["token"] for e in got["up"]]))
        self.assertAlmostEqual(got["layer_frac"], 1.0)
        self.assert_clean(eng)

    def test_up_and_down_sorted_and_k(self):
        eng = self.make_eng(seed=2)
        got = reader.pushes_toward(eng, 0, 0, k=6)
        self.assertEqual(len(got["up"]), 6)
        self.assertEqual(len(got["down"]), 6)
        up_logits = [e["logit"] for e in got["up"]]
        down_logits = [e["logit"] for e in got["down"]]
        self.assertTrue(all(a >= b for a, b in
                            zip(up_logits, up_logits[1:])))
        self.assertTrue(all(a <= b for a, b in
                            zip(down_logits, down_logits[1:])))
        self.assertEqual(len({e["id"] for e in got["up"]}), 6)
        self.assertEqual(len({e["id"] for e in got["down"]}), 6)
        big = reader.pushes_toward(eng, 0, 0, k=100)
        self.assertEqual(len(big["up"]), len(VOCAB))
        self.assertEqual(len(big["down"]), len(VOCAB))
        with self.assertRaises(ValueError):
            reader.pushes_toward(eng, 0, 0, k=0)
        with self.assertRaises(ValueError):
            reader.pushes_toward(eng, 5, 0)
        with self.assertRaises(ValueError):
            reader.pushes_toward(eng, 0, 99)
        self.assert_clean(eng)


class CoherenceTest(ReaderCase):
    def test_coherence_examples(self):
        self.assertEqual(
            reader.coherence(["pirate", "ship", "sea"]), 1.0)
        self.assertEqual(
            reader.coherence(["▁ing", "##s", "x"]), 0.0)
        self.assertEqual(
            reader.coherence(["Pirate", "pirate", "PIRATE", "##s"]), 0.5)
        self.assertEqual(reader.coherence([]), 0.0)
        self.assertEqual(
            reader.coherence([" pirate", " ship"]), 1.0)


class FiresOnTest(ReaderCase):
    def test_fires_on_cache_round_trip(self):
        eng = self.make_eng(seed=0)
        pirate, ship = VOCAB.index("pirate"), VOCAB.index("ship")
        with torch.no_grad():
            emb = eng.model.embed.weight
            emb.zero_()
            emb[pirate] = torch.tensor([1., 0, 0, 0])
            emb[ship] = torch.tensor([0.5, 0, 0, 0])
            up0 = eng.model.model.layers[0].mlp.up_proj.weight
            up0.zero_()
            up0[2] = torch.tensor([1., 0, 0, 0])
            eng.model.model.layers[1].mlp.up_proj.weight.zero_()
        docs = ["pirate ship", "sea pirate", "ship sea"]
        cache = reader.build_fires_on_cache(
            eng, docs, self.tmpdir(), layers=[0, 1], k=4)
        with np.load(cache / "fires_on.npz") as data:
            act, pos = data["act"], data["pos"]
        self.assertEqual(act.shape, (2, 8, 4))
        self.assertEqual(act.dtype, np.float16)
        self.assertEqual(pos.dtype, np.int32)
        sample = json.loads((cache / "sample.json").read_text())
        self.assertEqual(sample["doc_starts"], [0, 3, 6])
        self.assertEqual(len(sample["tokens"]), 9)
        self.assertEqual(sample["model_id"], "fake-reader")
        got = reader.fires_on(cache, 0, 2, k=4, window=1)
        self.assertEqual(len(got), 4)
        self.assertAlmostEqual(got[0]["activation"], 1.0)
        self.assertEqual(got[0]["token"], "pirate")
        acts = [e["activation"] for e in got]
        self.assertTrue(all(a >= b for a, b in zip(acts, acts[1:])))
        for entry in got:
            start, end = entry["mark"]
            self.assertEqual(entry["context"][start:end], entry["token"])
        ones = [e for e in got if e["activation"] == 1.0]
        self.assertEqual(len(ones), 2)
        self.assertTrue(all(e["token"] == "pirate" for e in ones))
        self.assertEqual(len(reader.fires_on(cache, 0, 2, k=2)), 2)
        self.assert_clean(eng)

    def test_fires_on_skips_bos(self):
        eng = self.make_eng(seed=0)
        with torch.no_grad():
            emb = eng.model.embed.weight
            emb.zero_()
            emb[0] = torch.tensor([100., 0, 0, 0])
            for layer in (0, 1):
                up = eng.model.model.layers[layer].mlp.up_proj.weight
                up.zero_()
                up[0] = torch.tensor([1., 0, 0, 0])
        docs = ["pirate ship sea", "ship sea pirate"]
        cache = reader.build_fires_on_cache(eng, docs, self.tmpdir(), k=4)
        with np.load(cache / "fires_on.npz") as data:
            pos = data["pos"]
        sample = json.loads((cache / "sample.json").read_text())
        self.assertEqual(sample["doc_starts"], [0, 4])
        self.assertFalse(any(int(p) in (0, 4) for p in pos.reshape(-1)))
        for layer in (0, 1):
            for neuron in range(8):
                for entry in reader.fires_on(cache, layer, neuron, k=4):
                    self.assertNotEqual(entry["token"], "<bos>")
                    self.assertLess(entry["activation"], 50.0)
        self.assert_clean(eng)


class FindNeuronsTest(ReaderCase):
    def test_find_neurons_ranks_planted_neuron_first(self):
        eng = self.make_eng(seed=3)
        model = eng.model
        star = VOCAB.index("pirate")
        with torch.no_grad():
            head = model.lm_head.weight
            head.zero_()
            head[star] = torch.tensor([2., 0, 0, 0])
            head[:, 1:] = torch.randn(len(VOCAB), 3) * 0.1
            head[star, 1:] = torch.tensor([0., 0, 0])
            for layer in (0, 1):
                down = model.model.layers[layer].mlp.down_proj.weight
                down.zero_()
                down[:, :] = 0.01
            model.model.layers[1].mlp.down_proj.weight[:, 5] = (
                3.0 * head[star])
        got = reader.find_neurons(eng, "pirate", k=5)
        self.assertEqual(got["word"], "pirate")
        self.assertEqual(got["token_ids"], [star])
        self.assertEqual(len(got["results"]), 5)
        first = got["results"][0]
        self.assertEqual((first["layer"], first["neuron"]), (1, 5))
        self.assertGreater(first["score"], 2.0)
        self.assertEqual(first["rank"], 1)
        with self.assertRaises(ValueError):
            reader.find_neurons(eng, "   ")
        self.assert_clean(eng)


class WordNeuronsTest(ReaderCase):
    def test_word_neurons_returns_planted_neuron(self):
        eng = self.make_eng(seed=4)
        model = eng.model
        star = VOCAB.index("gold")
        with torch.no_grad():
            model.embed.weight.zero_()
            model.embed.weight[star] = torch.tensor([1., 0, 0, 0])
            model.model.layers[0].mlp.up_proj.weight.zero_()
            model.model.layers[0].mlp.up_proj.weight[6] = torch.tensor(
                [1., 0, 0, 0])
            model.model.layers[1].mlp.up_proj.weight.zero_()
        prompt = "ship gold sea"
        wrapped = steer.wrap_prompt_notk(eng.tokenizer, prompt, eng.spec)
        ids = eng.tokenizer(
            wrapped, return_tensors="pt")["input_ids"][0].tolist()
        at = ids.index(star)
        got = reader.word_neurons(
            eng, prompt, rows=[], run="base", token_index=at,
            medians={}, k=5)
        self.assertEqual(got["token"], "gold")
        self.assertEqual(got["token_index"], at)
        self.assertEqual(len(got["results"]), 5)
        first = got["results"][0]
        self.assertEqual((first["layer"], first["neuron"]), (0, 6))
        self.assertAlmostEqual(first["activation"], 1.0)
        self.assertGreater(first["z"], 2.0)
        zs = [r["z"] for r in got["results"]]
        self.assertTrue(all(a >= b for a, b in zip(zs, zs[1:])))
        again = reader.word_neurons(
            eng, prompt, rows=[], run="steered", token_index=at,
            medians={}, k=5)
        self.assertEqual(again["results"], got["results"])
        with self.assertRaises(ValueError):
            reader.word_neurons(
                eng, prompt, [], "sideways", at, {}, k=5)
        with self.assertRaises(ValueError):
            reader.word_neurons(eng, prompt, [], "base", 0, {}, k=5)
        with self.assertRaises(ValueError):
            reader.word_neurons(eng, prompt, [], "base", 99, {}, k=5)
        self.assert_clean(eng)


class SteerLiftTest(ReaderCase):
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
            head[star] = torch.tensor([0., 1.5, 0, 0])
            down0 = model.model.layers[0].mlp.down_proj.weight
            down0.zero_()
            down0[dim, 4] = 0.8
            model.model.layers[1].mlp.up_proj.weight.zero_()

    def test_steer_lift_above_one_for_planted_neuron(self):
        eng = self.make_eng(seed=5)
        self._plant(eng)
        got = reader.steer_lift(
            eng, 0, 4, 2.0, ["rum"], ["sea storm", "ship map"], {0: 1.0})
        self.assertEqual(got["words"], ["rum"])
        self.assertEqual(len(got["base_p"]), 1)
        self.assertEqual(len(got["steered_p"]), 1)
        self.assertGreater(got["steered_p"][0], got["base_p"][0])
        self.assertGreater(got["lift"], 1.0)
        self.assert_clean(eng)

    def test_steer_lift_equals_one_at_zero_dose(self):
        eng = self.make_eng(seed=5)
        self._plant(eng)
        got = reader.steer_lift(
            eng, 0, 4, 0.0, ["rum"], ["sea storm", "ship map"], {0: 1.0})
        self.assertAlmostEqual(got["lift"], 1.0)
        self.assertAlmostEqual(got["base_p"][0], got["steered_p"][0])
        with self.assertRaises(ValueError):
            reader.steer_lift(eng, 0, 4, 1.0, [], ["sea"], {0: 1.0})
        with self.assertRaises(ValueError):
            reader.steer_lift(eng, 0, 4, 1.0, ["rum"], [], {0: 1.0})
        self.assert_clean(eng)


class NoHooksTest(ReaderCase):
    def test_reader_leaves_no_hooks(self):
        eng = self.make_eng(seed=6)
        cache = self.tmpdir() / "cache"
        reader.build_fires_on_cache(
            eng, ["pirate ship", "sea storm"], cache, layers=[0], k=2)
        calls = [
            lambda: reader.pushes_toward(eng, 0, 1, k=4),
            lambda: reader.coherence(["pirate"]),
            lambda: reader.fires_on(cache, 0, 1, k=2),
            lambda: reader.find_neurons(eng, "pirate", layers=[0, 1], k=3),
            lambda: reader.word_neurons(
                eng, "ship sea", [], "base", 2, {}, k=3),
            lambda: reader.steer_lift(
                eng, 0, 1, 0.5, ["sea"], ["ship rum"], {0: 1.0}),
            lambda: reader.build_fires_on_cache(
                eng, ["gold map"], self.tmpdir(), layers=[1], k=2),
        ]
        for call in calls:
            call()
            self.assert_clean(eng)
        bad = [
            lambda: reader.pushes_toward(eng, 99, 0),
            lambda: reader.find_neurons(eng, "   "),
            lambda: reader.word_neurons(
                eng, "ship sea", [], "base", 99, {}, k=3),
            lambda: reader.word_neurons(
                eng, "ship sea", [], "sideways", 2, {}, k=3),
            lambda: reader.steer_lift(
                eng, 0, 1, 0.5, [], ["ship"], {0: 1.0}),
            lambda: reader.build_fires_on_cache(eng, [], self.tmpdir()),
            lambda: reader.fires_on(self.tmpdir() / "missing", 0, 0),
        ]
        for call in bad:
            with self.assertRaises(Exception):
                call()
            self.assert_clean(eng)
        eng2 = self.make_eng(seed=7)

        def _boom(ids):
            raise RuntimeError("boom")

        eng2.model.forward = _boom
        with self.assertRaises(RuntimeError):
            reader.build_fires_on_cache(
                eng2, ["pirate ship"], self.tmpdir(), layers=[0], k=2)
        self.assert_clean(eng2)


if __name__ == "__main__":
    unittest.main()
