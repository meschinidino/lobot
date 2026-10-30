"""T3 load proof: tokenizer + chat template + rung-1 model load + tiny generate.

Run (downloads ~8.7 GiB on first run, cached after):
  uv run python scripts/verify_load.py            # full proof (slow first time)
  uv run python scripts/verify_load.py --quick   # tokenizer + chat template only (fast)

Writes scripts/../logs/load_proof.json with the full receipt on success.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from huggingface_hub import HfApi

from model_lobotomy import steer

ROOT = Path(__file__).resolve().parent.parent
PROBE_PROMPT = "Say the word hello and nothing else."


def main(quick: bool) -> int:
    receipt: dict = {
        "model_id": steer.MODEL_ID,
        "torch": torch.__version__,
        "device": steer.DEVICE,
        "dtype": str(steer.DTYPE),
        "trust_remote_code": steer.TRUST_REMOTE_CODE,
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    info = HfApi().model_info(steer.MODEL_ID)
    receipt["revision_sha"] = info.sha
    receipt["gated"] = info.gated
    print(f"model: {steer.MODEL_ID} @ {info.sha[:12]} (gated={info.gated})")

    tok = steer.load_tokenizer()
    wrapped = steer.wrap_prompt(tok, PROBE_PROMPT)
    ids = tok(wrapped, return_tensors="pt")
    receipt["chat_template_ok"] = True
    receipt["probe_input_tokens"] = int(ids["input_ids"].numel())
    print(f"chat template ok, probe input tokens: {receipt['probe_input_tokens']}")
    if quick:
        print("QUICK OK (tokenizer + chat template).")
        return 0

    t0 = time.perf_counter()
    model = steer.load_model()
    receipt["load_seconds"] = round(time.perf_counter() - t0, 1)
    n_params = sum(p.numel() for p in model.parameters())
    receipt["params"] = n_params
    receipt["mps_driver_mem_after_load"] = torch.mps.driver_allocated_memory()
    print(f"loaded {n_params/1e9:.2f}B params in {receipt['load_seconds']}s")

    model.generation_config.do_sample = False  # greedy (v1 runner rule, D6/D7)
    input_ids = ids["input_ids"].to(steer.DEVICE)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=steer.MAX_NEW_TOKENS_PROBE, do_sample=False)
    receipt["gen_seconds"] = round(time.perf_counter() - t0, 1)
    receipt["peak_mps_driver_mem"] = torch.mps.driver_allocated_memory()
    new_tokens = out[0][input_ids.shape[1]:]
    receipt["generated_text"] = tok.decode(new_tokens, skip_special_tokens=True)
    receipt["new_tokens"] = len(new_tokens)
    print(f"generated ({receipt['new_tokens']} tokens, {receipt['gen_seconds']}s): {receipt['generated_text']!r}")

    logdir = ROOT / "logs"
    logdir.mkdir(exist_ok=True)
    (logdir / "load_proof.json").write_text(json.dumps(receipt, indent=2))
    print(f"receipt -> logs/load_proof.json")
    print("LOAD PROOF OK: rung-1 weights load and generate on MPS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--quick" in sys.argv))
