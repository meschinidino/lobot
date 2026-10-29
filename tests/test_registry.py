"""Unit tests: multi-model registry (Qwen + Llama).

Qwen stays the default spec, so all existing no-spec calls keep working.
Llama 3.2 3B uses the plain LLaMA block path model.model.layers[L] with
dims 28/3072/8192. No weights: fakes only.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import steer  # noqa: E402
from steer import RecipeRow  # noqa: E402


def make_llama_fake(layers=(13, 14)):
    blocks = {}
    for layer in layers:
        lin = torch.nn.Linear(8192, 3072)
        blocks[layer] = SimpleNamespace(mlp=SimpleNamespace(down_proj=lin))
    return SimpleNamespace(
        model=SimpleNamespace(layers=blocks),
        config=SimpleNamespace(num_hidden_layers=28, hidden_size=3072,
                               intermediate_size=8192),
    )


def make_qwen_fake():
    lin = torch.nn.Linear(9216, 2560)
    inner = SimpleNamespace(
        layers={14: SimpleNamespace(mlp=SimpleNamespace(down_proj=lin))})
    return SimpleNamespace(
        model=SimpleNamespace(language_model=inner),
        config=SimpleNamespace(num_hidden_layers=32, hidden_size=2560,
                               intermediate_size=9216),
    )


class RegistryTest(unittest.TestCase):
    def test_specs_registered(self):
        self.assertEqual(steer.get_spec(steer.MODEL_ID), steer.QWEN_SPEC)
        self.assertEqual(steer.get_spec(steer.LLAMA_MODEL_ID), steer.LLAMA_SPEC)

    def test_unknown_model_fails_loudly(self):
        with self.assertRaises(RuntimeError):
            steer.get_spec("someone/else-7B")

    def test_model_class_dispatch(self):
        from transformers import LlamaForCausalLM, Qwen3_5ForConditionalGeneration
        self.assertIs(steer.model_class_for(steer.MODEL_ID),
                      Qwen3_5ForConditionalGeneration)
        self.assertIs(steer.model_class_for(steer.LLAMA_MODEL_ID),
                      LlamaForCausalLM)
        self.assertIs(steer.model_class_for("Qwen/Qwen3.5-2B"),
                      Qwen3_5ForConditionalGeneration)

    def test_calibration_paths_split(self):
        self.assertEqual(steer.calibration_path_for(steer.MODEL_ID),
                         "calibration.json")
        self.assertEqual(
            steer.calibration_path_for(steer.LLAMA_MODEL_ID),
            "calibration.meta-llama-llama-3.2-3b-instruct.json")


class LlamaPathTest(unittest.TestCase):
    def test_llama_block_resolves(self):
        model = make_llama_fake()
        mod = steer.resolve_down_proj(model, 14, steer.LLAMA_SPEC)
        self.assertIsInstance(mod, torch.nn.Linear)
        self.assertEqual(tuple(mod.weight.shape), (3072, 8192))

    def test_qwen_spec_rejects_llama_layout(self):
        with self.assertRaises(RuntimeError):
            steer.resolve_down_proj(make_llama_fake(), 14, steer.QWEN_SPEC)

    def test_llama_spec_rejects_qwen_layout(self):
        with self.assertRaises(RuntimeError):
            steer.resolve_down_proj(make_qwen_fake(), 14, steer.LLAMA_SPEC)

    def test_llama_bounds(self):
        steer.validate_recipe([RecipeRow(27, 8191, 1.0)], steer.LLAMA_SPEC)
        with self.assertRaises(ValueError):
            steer.validate_recipe([RecipeRow(28, 0, 1.0)], steer.LLAMA_SPEC)
        with self.assertRaises(ValueError):
            steer.validate_recipe([RecipeRow(0, 8192, 1.0)], steer.LLAMA_SPEC)

    def test_qwen_bounds_unchanged(self):
        steer.validate_recipe([RecipeRow(31, 9215, 1.0)])
        with self.assertRaises(ValueError):
            steer.validate_recipe([RecipeRow(32, 0, 1.0)])

    def test_llama_dims_assert(self):
        steer.assert_text_dims(make_llama_fake(), steer.LLAMA_SPEC)
        with self.assertRaises(RuntimeError):
            steer.assert_text_dims(make_llama_fake(), steer.QWEN_SPEC)
        with self.assertRaises(RuntimeError):
            steer.assert_text_dims(make_qwen_fake(), steer.LLAMA_SPEC)

    def test_recipe_string_template_per_model(self):
        rows = [RecipeRow(14, 100, 1.0)]
        qwen = steer.recipe_string(rows, "abc", max_new_tokens=150, seed=0,
                                   calib_prompts=["x"])
        llama = steer.recipe_string(rows, "abc", max_new_tokens=150, seed=0,
                                    calib_prompts=["x"],
                                    model_id=steer.LLAMA_MODEL_ID)
        self.assertIn("template=chat+no-think", qwen)
        self.assertIn("template=chat ", llama)
        self.assertIn(f"model={steer.LLAMA_MODEL_ID}", llama)


if __name__ == "__main__":
    unittest.main()
