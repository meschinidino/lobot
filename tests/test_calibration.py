"""Unit tests: calibration cache key, guards, median on fixture (T8).

Doc (D4/D9): per model+layer medians over tokens 1: of the wrapped prefill;
tap = decoder block output; each prompt yields 2-plus tokens (assert +
filter empties); steering hooks stay OFF during calibration; cache keyed
by model id + revision + template + dtype/quant + device + cal code.
Fake tokenizer/model fixtures, no weights.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import steer  # noqa: E402


class FakeIds:
    def __init__(self, n_tokens):
        self.n_tokens = n_tokens

    def numel(self):
        return self.n_tokens

    def to(self, device):
        return self


class FakeTok:
    chat_template = "fake-template-v1"

    def __init__(self, n_tokens=8):
        self.n_tokens = n_tokens

    def apply_chat_template(self, *a, **k):
        return "wrapped prompt text"

    def __call__(self, *a, **k):
        return {"input_ids": FakeIds(self.n_tokens)}


class FakeBlock:
    def __init__(self):
        self.fn = None
        self.removed = False

    def register_forward_hook(self, fn):
        self.fn = fn
        outer = self

        class _Handle:
            def remove(self):
                outer.removed = True

        return _Handle()


class FakeModel:
    """Layer blocks whose fixture hidden states feed the median math."""

    def __init__(self, layers, fixtures):
        self.blocks = {layer: FakeBlock() for layer in layers}
        self.model = SimpleNamespace(
            language_model=SimpleNamespace(layers=self.blocks))
        self.fixtures = list(fixtures)
        self.calls = 0

    def __call__(self, ids):
        self.calls += 1
        hidden = self.fixtures[min(self.calls - 1, len(self.fixtures) - 1)]
        for block in self.blocks.values():
            block.fn(block, (ids,), hidden)
        return None


def hidden_with_norms(token_norms, hidden=8):
    """[1, T, hidden] tensor whose per-token norms equal token_norms."""
    rows = []
    for v in token_norms:
        row = torch.zeros(hidden)
        row[0] = v
        rows.append(row)
    return torch.stack(rows).unsqueeze(0)


class CacheKeyTest(unittest.TestCase):
    def test_d9_fields_present(self):
        key = steer.calibration_cache_key("m", "r", FakeTok(),
                                          "torch.float16", "mps")
        for field in ("model_id", "revision", "template_sha256", "thinking",
                      "dtype", "quant", "device", "cal_code"):
            self.assertIn(field, key)
        self.assertIsNone(key["quant"])
        self.assertFalse(key["thinking"])
        self.assertEqual(len(key["template_sha256"]), 16)

    def test_template_change_changes_key(self):
        a = steer.calibration_cache_key("m", "r", FakeTok(),
                                        "torch.float16", "mps")

        class _OtherTok(FakeTok):
            chat_template = "other-template"

        b = steer.calibration_cache_key("m", "r", _OtherTok(),
                                        "torch.float16", "mps")
        self.assertNotEqual(a["template_sha256"], b["template_sha256"])

    def test_quant_rung_keyed(self):
        key = steer.calibration_cache_key("m", "r", FakeTok(),
                                          "torch.float16", "mps",
                                          quant="fp8")
        self.assertEqual(key["quant"], "fp8")


class GuardTest(unittest.TestCase):
    def test_all_empty_prompts_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            steer.calibrate_layer_medians(
                None, FakeTok(), ["", "   "], [13],
                cache_path=Path("/nonexistent/c.json"), key={})

    def test_single_token_prompt_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "<2 tokens"):
                steer.calibrate_layer_medians(
                    FakeModel([13], [hidden_with_norms([1.0])]),
                    FakeTok(n_tokens=1), ["hello"], [13],
                    cache_path=Path(td) / "c.json", key={"k": 1})

    def test_hooks_on_rejected_before_model_use(self):
        # model=None proves the assert fires before any model access.
        steer._active_steering.append(object())
        try:
            with self.assertRaisesRegex(RuntimeError, "OFF"):
                steer.calibrate_layer_medians(
                    None, FakeTok(), ["hello world"], [13],
                    cache_path=Path("/nonexistent/c.json"), key={})
        finally:
            steer._active_steering.pop()


class MedianFixtureTest(unittest.TestCase):
    def test_median_over_tokens_1_skips_bos(self):
        fixture = hidden_with_norms([1000.0, 3.0, 4.0, 5.0])
        with tempfile.TemporaryDirectory() as td:
            med, info = steer.calibrate_layer_medians(
                FakeModel([13], [fixture]), FakeTok(), ["some prompt"],
                [13], cache_path=Path(td) / "c.json", key={"k": 1})
        self.assertFalse(info["cache_hit"])
        self.assertAlmostEqual(med[13], 4.0, places=5)

    def test_token_counts_recorded_per_prompt(self):
        fixture = hidden_with_norms([1000.0, 3.0, 4.0, 5.0])
        with tempfile.TemporaryDirectory() as td:
            cp = Path(td) / "c.json"
            steer.calibrate_layer_medians(
                FakeModel([13], [fixture]), FakeTok(), ["prompt a here"],
                [13], cache_path=cp, key={"k": 1})
            payload = json.loads(cp.read_text())
        self.assertEqual(payload["token_counts"]["13"], [3])
        self.assertEqual(len(payload["prompt_hashes"]), 1)

    def test_capture_hooks_removed_afterwards(self):
        fixture = hidden_with_norms([1000.0, 3.0, 4.0, 5.0])
        model = FakeModel([13, 14], [fixture])
        with tempfile.TemporaryDirectory() as td:
            steer.calibrate_layer_medians(
                model, FakeTok(), ["prompt a here"], [13, 14],
                cache_path=Path(td) / "c.json", key={"k": 1})
        self.assertTrue(all(b.removed for b in model.blocks.values()))

    def test_median_pools_tokens_across_prompts(self):
        # Doc: "run three fixed neutral calibration prompts ... take the
        # median norm over tokens 1:". Pooled norms here are
        # [100, 101, 1, 2, 3, 4, 5] -> median 4. Last-prompt-only pooling
        # gives 3 (regression pinned 2026-10-29, fixed same day).
        fix1 = hidden_with_norms([999.0, 100.0, 101.0])
        fix2 = hidden_with_norms([999.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        with tempfile.TemporaryDirectory() as td:
            med, _ = steer.calibrate_layer_medians(
                FakeModel([13], [fix1, fix2]), FakeTok(),
                ["first prompt here", "second prompt here"], [13],
                cache_path=Path(td) / "c.json", key={"k": 1})
        self.assertAlmostEqual(med[13], 4.0, places=5)


class CacheRoundTripTest(unittest.TestCase):
    def _payload(self, prompts, key, medians):
        full = dict(key)
        full["prompts_sha256"] = hashlib.sha256(
            "\n".join(prompts).encode()).hexdigest()[:16]
        return {"key": full, "medians": medians, "prompt_hashes": ["h"],
                "token_counts": {"13": [3]}}

    def test_hit_reuses_medians_without_model(self):
        prompts = ["a prompt here"]
        key = {"model_id": "m"}
        payload = self._payload(prompts, key, {"13": 9.5})
        with tempfile.TemporaryDirectory() as td:
            cp = Path(td) / "c.json"
            cp.write_text(json.dumps(payload))
            med, info = steer.calibrate_layer_medians(
                None, FakeTok(), prompts, [13], cache_path=cp,
                key=dict(key))
        self.assertTrue(info["cache_hit"])
        self.assertEqual(med, {13: 9.5})

    def test_miss_recomputes_and_writes_cache(self):
        fixture = hidden_with_norms([999.0, 3.0, 4.0, 5.0])
        with tempfile.TemporaryDirectory() as td:
            cp = Path(td) / "c.json"
            med, info = steer.calibrate_layer_medians(
                FakeModel([13], [fixture]), FakeTok(), ["prompt a here"],
                [13], cache_path=cp, key={"model_id": "m"})
            self.assertFalse(info["cache_hit"])
            payload = json.loads(cp.read_text())
            self.assertIn("prompts_sha256", payload["key"])
            self.assertAlmostEqual(payload["medians"]["13"], 4.0, places=5)
            # Second call hits the just-written cache without a model.
            med2, info2 = steer.calibrate_layer_medians(
                None, FakeTok(), ["prompt a here"], [13], cache_path=cp,
                key={"model_id": "m"})
            self.assertTrue(info2["cache_hit"])
            self.assertEqual(med, med2)

    def test_missing_layer_in_cache_forces_recompute(self):
        prompts = ["a prompt here"]
        key = {"model_id": "m"}
        payload = self._payload(prompts, key, {"13": 9.5})  # no L14
        fixture = hidden_with_norms([999.0, 3.0, 4.0, 5.0])
        with tempfile.TemporaryDirectory() as td:
            cp = Path(td) / "c.json"
            cp.write_text(json.dumps(payload))
            med, info = steer.calibrate_layer_medians(
                FakeModel([13, 14], [fixture]), FakeTok(), prompts,
                [13, 14], cache_path=cp, key=dict(key))
        self.assertFalse(info["cache_hit"])
        self.assertIn(14, med)

    def test_recompute_merges_earlier_layers_same_key(self):
        # A subset run must not drop layers calibrated earlier (repo
        # clobber 2026-10-29: a layer-11-only run erased layers 10-17).
        prompts = ["a prompt here"]
        key = {"model_id": "m"}
        payload = self._payload(prompts, key, {"13": 9.5})
        fixture = hidden_with_norms([999.0, 3.0, 4.0, 5.0])
        with tempfile.TemporaryDirectory() as td:
            cp = Path(td) / "c.json"
            cp.write_text(json.dumps(payload))
            med, info = steer.calibrate_layer_medians(
                FakeModel([14], [fixture]), FakeTok(), prompts,
                [14], cache_path=cp, key=dict(key))
            stored = json.loads(cp.read_text())
        self.assertFalse(info["cache_hit"])
        self.assertAlmostEqual(med[14], 4.0, places=5)
        self.assertEqual(stored["medians"]["13"], 9.5)
        self.assertAlmostEqual(stored["medians"]["14"], 4.0, places=5)
        self.assertEqual(stored["token_counts"]["13"], [3])
        self.assertEqual(stored["token_counts"]["14"], [3])

    def test_recompute_drops_old_layers_on_key_change(self):
        # Different prompts: no merge, the stale layers go away.
        payload = self._payload(["old prompt here"], {"model_id": "m"},
                                {"13": 9.5})
        fixture = hidden_with_norms([999.0, 3.0, 4.0, 5.0])
        with tempfile.TemporaryDirectory() as td:
            cp = Path(td) / "c.json"
            cp.write_text(json.dumps(payload))
            steer.calibrate_layer_medians(
                FakeModel([14], [fixture]), FakeTok(), ["new prompt here"],
                [14], cache_path=cp, key={"model_id": "m"})
            stored = json.loads(cp.read_text())
        self.assertNotIn("13", stored["medians"])
        self.assertIn("14", stored["medians"])

    def test_stored_repo_cache_has_full_key(self):
        stored = json.loads((ROOT / "docs" / "calibration.json").read_text())
        for field in ("model_id", "revision", "template_sha256", "thinking",
                      "dtype", "quant", "device", "cal_code",
                      "prompts_sha256"):
            self.assertIn(field, stored["key"])
        self.assertGreater(stored["medians"]["13"], 0)
        self.assertGreater(stored["medians"]["14"], 0)
        self.assertEqual(len(stored["prompt_hashes"]), 3)
        for counts in stored["token_counts"].values():
            self.assertTrue(all(c >= 2 for c in counts))


if __name__ == "__main__":
    unittest.main()
