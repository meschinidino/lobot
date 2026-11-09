"""Single-prefill generation: no timed extra forward pass. No weights."""
import unittest

import torch

from model_lobotomy import steer


class FakeTok:
    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


class FakeModel:
    def __init__(self):
        self.forward_calls = 0
        self._p = torch.zeros(1)

    def parameters(self):
        return iter([self._p])

    def __call__(self, input_ids):
        self.forward_calls += 1
        raise AssertionError("extra prefill forward must not run")

    def generate(self, input_ids, max_new_tokens, do_sample,
                 stopping_criteria=None, streamer=None,
                 logits_processor=None):
        out = input_ids.clone()
        for step in range(max_new_tokens):
            out = torch.cat(
                [out, torch.tensor([[10 + step]])], dim=1)
            if logits_processor is not None:
                logits_processor(out, torch.zeros(1, 4))
            if stopping_criteria is not None:
                stop = stopping_criteria(out, None)
                if bool(stop.all()):
                    break
        return out


class SinglePrefillTest(unittest.TestCase):
    def test_generate_text_runs_no_extra_forward(self):
        model = FakeModel()
        result = steer.generate_text(
            model, FakeTok(), "hi", max_new_tokens=3, seed=0,
            wrapped="hi")
        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(result["token_ids"], [10, 11, 12])
        self.assertEqual(result["new_tokens"], 3)
        self.assertGreaterEqual(result["prefill_s"], 0.0)
        self.assertLessEqual(result["prefill_s"],
                             result["seconds"] + 0.05)
        self.assertGreaterEqual(result["decode_s"], 0.0)

    def test_generate_text_with_streamer_runs_no_extra_forward(self):
        model = FakeModel()
        result = steer.generate_text_with_streamer(
            model, FakeTok(), "hi", max_new_tokens=2, seed=0,
            wrapped="hi")
        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(result["token_ids"], [10, 11])

    def test_first_token_processor_records_once(self):
        proc = steer.FirstTokenProcessor()
        ids = torch.tensor([[1, 2]])
        scores = torch.zeros(1, 4)
        self.assertIs(proc(ids, scores), scores)
        self.assertIsNotNone(proc.first_step_at)
        first = proc.first_step_at
        proc(ids, scores)
        self.assertEqual(proc.first_step_at, first)

    def test_measure_prefill_false_reports_none_not_zero(self):
        model = FakeModel()
        result = steer.generate_text(
            model, FakeTok(), "hi", max_new_tokens=2, seed=0,
            wrapped="hi", measure_prefill=False)
        self.assertIsNone(result["prefill_s"])
        self.assertGreaterEqual(result["decode_s"], 0.0)


class DialIndexTest(unittest.TestCase):
    def test_token_k_reads_tap_k(self):
        from model_lobotomy import engine as engine_mod
        unit = torch.tensor([1.0, 0.0])
        tap = [torch.tensor([0.0, 0.0]), torch.tensor([5.0, 0.0]),
               torch.tensor([10.0, 0.0])]
        # b0=0 (neutral), b1=10 (trait): token0 -> 0.0, token1 -> 0.5
        self.assertEqual(engine_mod.dial_for_index(tap, 0, unit, 0, 10), 0.0)
        self.assertEqual(engine_mod.dial_for_index(tap, 1, unit, 0, 10), 0.5)
        self.assertEqual(engine_mod.dial_for_index(tap, 2, unit, 0, 10), 1.0)
        self.assertIsNone(engine_mod.dial_for_index(tap, 3, unit, 0, 10))


if __name__ == "__main__":
    unittest.main()
