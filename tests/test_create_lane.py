"""Create lane tests: fake engine and fake daemon only, no weights.

These prove the batching plumbing (TorchLane keeps today's calls,
LlamaLane batches and converts judge/NLL the same way, CreateDaemon
chunks, create_job opts in by env). Real-weight equality is the
model lane (docs/plans/llama-backend.md, Create lane).
"""
from __future__ import annotations

import math
import os
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_lobotomy import create_job  # noqa: E402
from model_lobotomy import engine as engine_mod  # noqa: E402
from model_lobotomy import llama_backend as lb  # noqa: E402
from model_lobotomy import steer  # noqa: E402
from model_lobotomy import trait_eval as T  # noqa: E402

VEC = engine_mod.EditSet(vector=(
    engine_mod.VectorEdit(14, tuple([1.0] + [0.0] * 2559), 1.0, 1.0),))
ABL = engine_mod.EditSet(ablate=(
    engine_mod.Ablation(14, tuple([1.0] + [0.0] * 2559)),))


class FakeEng:
    """Torch-engine stand-in: reply text encodes (prompt, edits)."""

    def __init__(self):
        self.calls = []
        self.model = object()
        self.tokenizer = object()

    def run(self, prompt, edits, medians, max_new_tokens, seed,
            cancel=None):
        self.calls.append((prompt, edits))
        tag = ("base" if edits == engine_mod.EMPTY_EDITS
               else "abl" if edits.ablate else "steered")
        return {"text": f"{tag}:{prompt}"}


class FakeTok:
    def __call__(self, text, add_special_tokens=True,
                 return_tensors=None):
        ids = [3 + (ord(c) % 40) for c in text]
        return {"input_ids": ([1] if add_special_tokens else []) + ids}

    def decode(self, ids, skip_special_tokens=True):
        return "|".join(str(i) for i in ids if not (
            skip_special_tokens and i == 1))


class FakeDaemon:
    def __init__(self, last=(2.0, 0.0), lp=(-1.0, -3.0)):
        self.tokenizer = FakeTok()
        self.gens, self.scores = [], []
        self.last, self.lp = list(last), list(lp)

    def info(self):
        return {"quant": "f16"}

    def gen_ids_many(self, prompts, edits, max_new, cancel=None):
        self.gens.append((len(prompts), edits, max_new))
        return [[7, 8, 1] for _ in prompts]

    def score_many(self, items, cancel=None):
        self.scores.append(items)
        return [{"last": self.last if "last_ids" in it else [],
                 "lp": self.lp if "from" in it else []} for it in items]


def _wrap_plain(tok, text, spec=None):
    return f"<{text}>"


class TorchLaneTest(unittest.TestCase):
    def test_gen_many_calls_run_per_prompt(self):
        eng = FakeEng()
        lane = T.TorchLane(eng, None)
        out = lane.gen_many(["a", "b"], VEC, 24)
        self.assertEqual(out, ["steered:a", "steered:b"])
        self.assertEqual([c[0] for c in eng.calls], ["a", "b"])

    def test_judge_and_nll_memoized(self):
        lane = T.TorchLane(FakeEng(), None)
        with mock.patch.object(T, "judge_score",
                               return_value=0.7) as js, \
                mock.patch.object(T.spike, "nll",
                                  return_value=1.5) as nll:
            a = lane.judge_many([("q", "r"), ("q", "r")], "j")
            b = lane.judge_many([("q", "r")], "j")
            c = lane.nll_many([("q", "r"), ("q", "r")])
        self.assertEqual(a + b, [0.7, 0.7, 0.7])
        self.assertEqual(js.call_count, 1)
        self.assertEqual(c, [1.5, 1.5])
        self.assertEqual(nll.call_count, 1)

    def test_label_is_part_of_judge_key(self):
        lane = T.TorchLane(FakeEng(), None)
        with mock.patch.object(T, "judge_score",
                               side_effect=[0.1, 0.9]) as js:
            out = lane.judge_many([("q", "r")], "a") + lane.judge_many(
                [("q", "r")], "b")
        self.assertEqual(out, [0.1, 0.9])
        self.assertEqual(js.call_count, 2)


class LlamaLaneTest(unittest.TestCase):
    def setUp(self):
        p1 = mock.patch.object(T.steer, "wrap_prompt_notk", _wrap_plain)
        p2 = mock.patch.object(T.spike, "_yes_no_ids",
                               return_value=[5, 6])
        p1.start()
        p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        self.eng = FakeEng()
        self.daemon = FakeDaemon()
        self.lane = T.LlamaLane(self.eng, steer.QWEN_SPEC, self.daemon)

    def test_vector_and_empty_batch_on_daemon(self):
        out = self.lane.gen_many(["a", "b", "c"], VEC, 60)
        self.assertEqual(out, ["7|8"] * 3)  # stop id 1 dropped
        self.assertEqual(self.daemon.gens, [(3, VEC, 60)])
        self.lane.gen_many(["a"], engine_mod.EMPTY_EDITS, 60)
        self.assertEqual(self.eng.calls, [])

    def test_ablation_falls_back_to_torch(self):
        out = self.lane.gen_many(["a", "b"], ABL, 60)
        self.assertEqual(out, ["abl:a", "abl:b"])
        self.assertEqual(self.daemon.gens, [])

    def test_ablation_fallback_memoized(self):
        self.lane.gen_many(["a", "b"], ABL, 60)
        out = self.lane.gen_many(["b", "c"], ABL, 60)
        self.assertEqual(out, ["abl:b", "abl:c"])
        self.assertEqual([c[0] for c in self.eng.calls], ["a", "b", "c"])

    def test_judge_formula_and_refusal_gate(self):
        with mock.patch.object(T, "refusal_gate",
                               side_effect=lambda r: r == "no"):
            out = self.lane.judge_many([("q", "yes"), ("q", "no")], "j")
        self.assertAlmostEqual(out[0], 1 / (1 + math.exp(0.0 - 2.0)))
        self.assertEqual(out[1], 0.0)
        (items,) = self.daemon.scores
        self.assertEqual(len(items), 1)  # refusals never scored
        self.assertEqual(items[0]["last_ids"], [5, 6])
        tok = self.daemon.tokenizer
        want = tok(_wrap_plain(tok, T.judge_prompt("q", "yes", "j")))
        self.assertEqual(items[0]["ids"], want["input_ids"])

    def test_nll_is_mean_negative_logprob(self):
        out = self.lane.nll_many([("q", "rr"), ("q", "")])
        self.assertAlmostEqual(out[0], 2.0)
        self.assertTrue(math.isnan(out[1]))
        (items,) = self.daemon.scores
        tok = self.daemon.tokenizer
        ctx = tok(_wrap_plain(tok, "q"))["input_ids"]
        rep = tok("rr", add_special_tokens=False)["input_ids"]
        self.assertEqual(items[0], {"ids": ctx + rep, "from": len(ctx)})

    def test_info_names_lane(self):
        self.assertEqual(self.lane.info(), {"lane": "llama",
                                            "quant": "f16"})


class BatchedStagesTest(unittest.TestCase):
    """Batched stage paths give the same numbers as one-by-one."""

    def _judge(self, model, tok, spec, q, reply, label):
        return {"base": 0.2, "steered": 0.8, "abl": 0.1}[
            reply.split(":")[0]] + 0.01 * len(q)

    def test_verb_final_batched_equals_sequential(self):
        qs = [f"q{i}" for i in range(7)]
        with mock.patch.object(T, "judge_score", side_effect=self._judge):
            full = T.verb_final(FakeEng(), None, {"judge": "j"}, "j", qs,
                                VEC, ABL, max_new=24)
            seq = T.verb_final(FakeEng(), None, {"judge": "j"}, "j", qs,
                               VEC, ABL, n0=100, nmax=7, max_new=24)
        self.assertEqual(full["gains"], seq["gains"])
        self.assertEqual(full["control"], seq["control"])
        self.assertEqual(full["pairs"], seq["pairs"])

    def test_base_many_counts_like_base(self):
        eng = FakeEng()
        lane = T.TorchLane(eng, None)
        cache = T.BaseCache()
        a = cache.base_many(lane, ["x", "y", "x"], 24)
        b = cache.base_many(lane, ["y"], 24)
        self.assertEqual(a + b, ["base:x", "base:y", "base:x", "base:y"])
        self.assertEqual((cache.misses, cache.hits), (2, 2))
        self.assertEqual(len(eng.calls), 2)

    def test_tune_nll_min_pairs_plain_and_prefixed(self):
        import torch
        eng = FakeEng()
        lane = T.TorchLane(eng, None)
        nlls = {}

        def fake_nll(model, tok, spec, q, r):
            nlls[(q, r)] = 2.0 if q.startswith("POS") else 1.0
            return nlls[(q, r)]

        spec = steer.ModelSpec("t", 8, 4, 8, "llama", False)
        A = {L: torch.randn(6, 4) + 1 for L in range(8)}
        B = {L: torch.randn(6, 4) - 1 for L in range(8)}
        cfg = {"judge": "j", "pos": "POS"}
        params = dict(T.SCHEDULES["full"], tune_doses=[0.0, 1.0],
                      tune_top=1)
        with mock.patch.object(T, "judge_score", side_effect=self._judge), \
                mock.patch.object(T.spike, "nll", side_effect=fake_nll):
            _frozen, table, _ = T.tune_layer_dose(
                eng, spec, cfg, {L: 1.0 for L in range(8)}, A, B,
                ["a", "b"], "j", params=params, lane=lane)
        (row,) = table.values()
        self.assertEqual(row[0.0]["nll"], 1.0)
        self.assertEqual(row[1.0]["nll"], 1.0)
        self.assertIn(("POS\n\na", "steered:a"), nlls)


class FakeBatchProc:
    def __init__(self, binary, gguf, extra_args=None):
        self.extra_args = extra_args
        self.info = {"n_layer": 32, "n_embd": 2560}
        self.batches = []
        self.closed = False

    def generate_batch(self, prompts, max_new, stop_ids, cvec,
                       cancel=None):
        self.batches.append((len(prompts), max_new, cvec is not None))
        return {"outs": [[9] for _ in prompts], "stop": "maxlen"}

    def score_batch(self, items, cvec=None, cancel=None):
        return [{"lp": [], "last": []} for _ in items]

    def close(self):
        self.closed = True


class CreateDaemonTest(unittest.TestCase):
    def _daemon(self):
        with mock.patch.object(lb, "resolve_gguf",
                               return_value=("x.gguf", {"quant": "f16"})), \
                mock.patch.object(lb.LlamaEngine, "_resolve_stop_ids",
                                  return_value=[1]):
            return lb.CreateDaemon(steer.MODEL_ID, "rev", FakeTok(),
                                   proc_factory=FakeBatchProc)

    def test_parallel_args_and_chunks(self):
        d = self._daemon()
        self.assertIn("--parallel", d.proc.extra_args)
        self.assertIn("--no-prefix-cache", d.proc.extra_args)
        outs = d.gen_ids_many([[1, 2]] * 70, VEC, 60)
        self.assertEqual(len(outs), 70)
        self.assertEqual([b[0] for b in d.proc.batches], [32, 32, 6])
        self.assertTrue(all(b[2] for b in d.proc.batches))

    def test_empty_edits_send_no_cvec(self):
        d = self._daemon()
        d.gen_ids_many([[1]], engine_mod.EMPTY_EDITS, 24)
        self.assertEqual(d.proc.batches, [(1, 24, False)])

    def test_ablation_rejected(self):
        d = self._daemon()
        with self.assertRaises(lb.UnsupportedError):
            d.gen_ids_many([[1]], ABL, 24)

    def test_cancel_stops_chunks(self):
        d = self._daemon()
        d.proc.generate_batch = lambda *a, **k: {"outs": [[9]],
                                                 "stop": "cancel"}
        self.assertEqual(len(d.gen_ids_many([[1]] * 40, VEC, 8)), 1)


class CreateJobLaneTest(unittest.TestCase):
    def test_default_is_torch(self):
        runner = create_job.CreateRunner(jobs_dir=Path("/tmp/x"),
                                         engine=object())
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LOBOT_CREATE_LANE", None)
            self.assertIsNone(runner._lane(object(), "m", "r"))

    def test_env_llama_builds_and_reuses_daemon(self):
        runner = create_job.CreateRunner(jobs_dir=Path("/tmp/x"),
                                         engine=object())

        class Eng:
            tokenizer = FakeTok()
            spec = steer.QWEN_SPEC

        made = []

        def fake_daemon(model_id, revision, tok):
            made.append((model_id, revision))
            return FakeDaemon()

        with mock.patch.dict(os.environ, {"LOBOT_CREATE_LANE": "llama"}), \
                mock.patch.object(lb, "CreateDaemon", fake_daemon):
            a = runner._lane(Eng(), "m", "r")
            b = runner._lane(Eng(), "m", "r")
        self.assertIsInstance(a, T.LlamaLane)
        self.assertIs(a.daemon, b.daemon)
        self.assertEqual(made, [("m", "r")])


if __name__ == "__main__":
    unittest.main()
