"""Integration lane: hook-fired proof on the real model (T9).

Doc: D5 (model-backed integration lane, slow, opt-in per run) + eng-review
test plan critical paths: "hook fires" + "dose 0 output equals baseline
exactly". Value: protects=hook fires + idle hook is exact no-op;
fails_when=hook misregistered or leaks; no_claim=evil quality (manual eval).

Opt-in: skipped unless MODEL_LOBOTOMY_INTEGRATION=1. Default `unittest
discover` stays fast with no weights. Needs the committed calibration.json
(cache hit asserted; a miss fails loudly instead of recomputing).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import steer  # noqa: E402
from model_lobotomy.steer import RecipeRow  # noqa: E402

OPT_IN = os.environ.get("MODEL_LOBOTOMY_INTEGRATION") == "1"
CANARY_PROMPT = "Say hello in one short sentence."
SEED = 0
GEN_TOKENS = 32  # short: controls, not quality (T1 judged quality at 150)

PRESET = json.loads((ROOT / "presets" / "evil-qwen.json").read_text())
EVIL_ROWS = [RecipeRow(**r) for r in PRESET["rows"]]
LAYERS = sorted({r.layer for r in EVIL_ROWS})


def _dose_zero(rows: list[RecipeRow]) -> list[RecipeRow]:
    return [RecipeRow(r.layer, r.neuron, 0.0) for r in rows]


@unittest.skipUnless(OPT_IN, "integration lane needs MODEL_LOBOTOMY_INTEGRATION=1 "
                             "(slow, loads 4B weights)")
class SteeringLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model, cls.tok, cls.rung_n, cls.attempts = steer.load_with_ladder()
        if cls.rung_n != 1:
            raise unittest.SkipTest(f"integration lane pins rung 1, got {cls.rung_n}")
        steer.check_preset_binding(PRESET, steer.MODEL_ID)
        steer.assert_text_dims(cls.model)
        # Calibrate against a temp copy so a cache miss can never clobber
        # the committed calibration.json (miss fails loudly instead).
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="t9-calib-")
        tmp_cache = Path(cls._tmpdir.name) / "calibration.json"
        shutil.copyfile(ROOT / "docs" / "calibration.json", tmp_cache)
        from huggingface_hub import model_info

        info = model_info(steer.MODEL_ID)
        key = steer.calibration_cache_key(
            steer.MODEL_ID, info.sha, cls.tok, str(steer.DTYPE), steer.DEVICE)
        medians, cinfo = steer.calibrate_layer_medians(
            cls.model, cls.tok, PRESET["calibration_prompts"], LAYERS,
            cache_path=tmp_cache, key=key)
        if not cinfo["cache_hit"]:
            raise AssertionError(
                "calibration cache miss on the committed calibration.json; "
                "refusing to recompute (would rewrite medians)")
        cls.medians = medians

    @classmethod
    def tearDownClass(cls):
        steer.remove_steering_hooks()
        del cls.model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        cls._tmpdir.cleanup()
        super().tearDownClass()

    def tearDown(self):
        steer.remove_steering_hooks()
        steer.reset_hook_call_count()
        self.assertEqual(steer.steering_hook_count(), 0)

    def _forward_logits(self, text: str = CANARY_PROMPT):
        device = str(next(self.model.parameters()).device)
        ids = self.tok(text, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            return self.model(ids).logits

    def test_canary_hook_fires(self):
        n = steer.hooks_fire_probe(self.model, self.tok)
        self.assertGreaterEqual(n, 1, "canary pre-hook never fired on L0 down_proj")

    def test_steered_hidden_states_differ(self):
        with torch.no_grad():
            base = self._forward_logits()
        steer.reset_hook_call_count()
        steer.register_steering_hooks(self.model, EVIL_ROWS, self.medians, mode="add")
        try:
            with torch.no_grad():
                steered = self._forward_logits()
            calls = steer.hook_call_count()
        finally:
            steer.remove_steering_hooks()
        self.assertGreater(calls, 0, "steering hooks never fired during forward pass")
        self.assertFalse(torch.equal(base, steered),
                         "evil doses left hidden states unchanged (dead hook?)")

    def test_dose_zero_logits_bit_identical(self):
        with torch.no_grad():
            base = self._forward_logits()
        steer.register_steering_hooks(self.model, _dose_zero(EVIL_ROWS),
                                      self.medians, mode="add")
        try:
            with torch.no_grad():
                idle = self._forward_logits()
        finally:
            steer.remove_steering_hooks()
        self.assertTrue(torch.equal(base, idle), "dose-0 hooks changed logits")

    def test_dose_zero_generation_exact(self):
        base = steer.generate_text(self.model, self.tok, CANARY_PROMPT,
                                   max_new_tokens=GEN_TOKENS, seed=SEED)
        steer.register_steering_hooks(self.model, _dose_zero(EVIL_ROWS),
                                      self.medians, mode="add")
        try:
            idle = steer.generate_text(self.model, self.tok, CANARY_PROMPT,
                                       max_new_tokens=GEN_TOKENS, seed=SEED)
        finally:
            steer.remove_steering_hooks()
        self.assertEqual(idle["token_ids"], base["token_ids"],
                         "dose-0 generation diverged from baseline")
        self.assertEqual(idle["text"], base["text"])
        self.assertGreater(base["new_tokens"], 0, "baseline generated nothing")

    def test_baseline_greedy_deterministic(self):
        first = steer.generate_text(self.model, self.tok, CANARY_PROMPT,
                                    max_new_tokens=GEN_TOKENS, seed=SEED)
        second = steer.generate_text(self.model, self.tok, CANARY_PROMPT,
                                     max_new_tokens=GEN_TOKENS, seed=SEED)
        self.assertEqual(second["token_ids"], first["token_ids"],
                         "baseline-vs-baseline differs; MPS nondeterminism gates "
                         "interpretation (D7 control)")


if __name__ == "__main__":
    unittest.main()
