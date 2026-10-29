"""model-lobotomy engine.

T3 scope (this file, this task): load recipe only.
Hook/steering code lands in T1, the fallback ladder in T2, calibration in T4,
the module-path assert in T5. Do not add them here yet.

Load recipe (Eng D2 rung 1 + D3 + D9, verified by scripts/verify_load.py):
  model ......... Qwen/Qwen3.5-4B (public, ungated, safetensors, 2 shards)
  class ......... Qwen3_5ForConditionalGeneration (in transformers, no remote code)
  dtype ......... torch.float16 on MPS (config native bf16 noted, rung 1 says fp16)
  device ........ single MPS device, eager (plain .to("mps"), no accelerate dispatch)
  prompt wrap ... tokenizer.apply_chat_template (chat model, verified live at T3)
  weights ....... 9,319,737,856 param bytes (~8.7 GiB); disk budget 15 GiB (T3)
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION: str | None = None  # float on main; resolved sha recorded by verify_load.py
DTYPE = torch.float16
DEVICE = "mps"
TRUST_REMOTE_CODE = False
MAX_NEW_TOKENS_PROBE = 16


def load_tokenizer(model_id: str = MODEL_ID):
    """Load the text tokenizer and assert the chat template exists."""
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=TRUST_REMOTE_CODE)
    if tok.chat_template is None:
        raise RuntimeError(f"{model_id} has no chat template; T3 load recipe needs one")
    return tok


def wrap_prompt(tokenizer, user_text: str) -> str:
    """Wrap a user prompt with the model's chat template (v1 rule)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model(model_id: str = MODEL_ID, revision: str | None = MODEL_REVISION):
    """Load rung-1 weights: fp16, eager, onto MPS. No hooks here (T1 owns hooks)."""
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; rung 1 needs MPS (see README ladder)")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        revision=revision,
        dtype=DTYPE,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    model.to(DEVICE)
    model.eval()
    return model
