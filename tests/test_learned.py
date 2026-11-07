"""ML v3: learned steering vectors on a tiny model (no weights)."""
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from model_lobotomy import learned


def _tiny():
    cfg = LlamaConfig(vocab_size=64, hidden_size=32,
                      intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=4)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg)


def _seq(side, ans, pair=0):
    ids = torch.tensor([1, 2, 3] + ans)
    return {"input_ids": ids, "answer_start": 3, "side": side,
            "pair": pair}


class FewShotPromptTest(unittest.TestCase):
    def test_no_shots_is_the_plain_prompt(self):
        self.assertEqual(learned.fewshot_prompt("Be rude.", [], "Why?"),
                         "Be rude.\n\nWhy?")

    def test_shots_show_the_style(self):
        out = learned.fewshot_prompt(
            "Be rude.", [{"q": "Nice hat?", "a": "It looks awful."}],
            "Why?")
        self.assertIn("Question: Nice hat?\nAnswer: It looks awful.", out)
        self.assertTrue(out.endswith("Question: Why?\nAnswer:"))


class ValidateVectorsTest(unittest.TestCase):
    def test_valid_vectors_pass(self):
        out = learned.validate_vectors(
            {3: torch.tensor([1.0, 0.0]), 7: torch.tensor([0.0, 2.0])},
            [3, 7])
        self.assertEqual(sorted(out), [3, 7])

    def test_zero_and_nan_and_missing_fail(self):
        with self.assertRaises(ValueError):
            learned.validate_vectors({3: torch.zeros(2)}, [3])
        with self.assertRaises(ValueError):
            learned.validate_vectors(
                {3: torch.tensor([float("nan"), 1.0])}, [3])
        with self.assertRaises(ValueError):
            learned.validate_vectors({3: torch.ones(2)}, [3, 7])
        with self.assertRaises(ValueError):
            learned.validate_vectors({3: torch.ones(2, 2)}, [3])


class TrainTest(unittest.TestCase):
    def setUp(self):
        self.model = _tiny()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mods = {L: self.model.model.layers[L] for L in (0, 1)}
        self.init = {L: torch.zeros(32) for L in (0, 1)}
        self.seqs = [s for i in range(4) for s in (_seq("pos", [5, 6, 7], i), _seq("neg", [8, 9, 10], i))]
        self.drift = [{"input_ids": torch.tensor([1, 2, 3, 4])}]

    def test_loss_falls_and_model_frozen(self):
        before = {n: p.clone() for n, p in
                  self.model.named_parameters()}
        vecs, hist = learned.train_vectors(
            self.model, self.mods, self.seqs, self.drift, self.init,
            epochs=6, batch_pairs=4, lr=0.1)
        self.assertGreater(hist[0]["pos_nll"], hist[-1]["pos_nll"])
        for n, p in self.model.named_parameters():
            self.assertTrue(torch.equal(before[n], p), n)
        self.assertTrue(any(not torch.equal(vecs[L], self.init[L])
                            for L in vecs))

    def test_hooks_removed_after_training(self):
        learned.train_vectors(self.model, self.mods, self.seqs,
                              self.drift, self.init, epochs=1,
                              batch_pairs=4)
        for m in self.mods.values():
            self.assertEqual(len(m._forward_hooks), 0)

    def test_empty_drift_rejected(self):
        with self.assertRaises(ValueError):
            learned.train_vectors(self.model, self.mods, self.seqs,
                                  [], self.init)

    def test_incomplete_pair_rejected(self):
        seqs = [_seq("pos", [5, 6, 7], 0)]
        with self.assertRaises(ValueError):
            learned.train_vectors(self.model, self.mods, seqs,
                                  self.drift, self.init)

    def test_empty_answer_rejected(self):
        seqs = [_seq("pos", [], 0), _seq("neg", [8], 0)]
        with self.assertRaises(ValueError):
            learned.train_vectors(self.model, self.mods, seqs,
                                  self.drift, self.init)

    def test_failed_registration_leaves_no_hooks(self):
        mods = {0: self.mods[0]}  # layer 1 missing on purpose
        with self.assertRaises(KeyError):
            learned.train_vectors(self.model, mods, self.seqs,
                                  self.drift, self.init, epochs=1)
        self.assertEqual(len(self.mods[0]._forward_hooks), 0)

    def test_drift_punishes_big_vector(self):
        big = {L: torch.full((32,), 3.0) for L in (0, 1)}
        _, hist_big = learned.train_vectors(
            self.model, self.mods, self.seqs, self.drift, big,
            epochs=1, batch_pairs=4, lr=0.0)
        _, hist_zero = learned.train_vectors(
            self.model, self.mods, self.seqs, self.drift, self.init,
            epochs=1, batch_pairs=4, lr=0.0)
        self.assertGreater(hist_big[0]["drift_kl"],
                           hist_zero[0]["drift_kl"])


if __name__ == "__main__":
    unittest.main()
