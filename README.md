# model-lobotomy

Visual neuron-steering workbench. Local-first, one model, before/after loop.
V1: load Qwen3.5-4B, pick layer/neuron/dose, run the same prompt with and
without steering, see the diff, copy a paste-ready artifact for X.

Design doc (source of truth, Status APPROVED):
`~/.gstack/projects/model-lobotomy/dinomeschini-unknown-design-20261028-233909.md`

## Setup (T3)

Requires `uv`. Python is pinned to 3.12 (Eng D3: torch/MPS wheels lag new
minors; system Python here is 3.14).

```sh
uv python install 3.12
uv sync
uv run python scripts/check_env.py      # import asserts + MPS + Qwen3.5 probe
uv run python scripts/verify_load.py    # full load proof (downloads ~8.7 GiB once)
```

Pinned versions (recorded 2026-10-29, see `uv.lock` for the full tree):

| piece      | version                                   |
|------------|-------------------------------------------|
| python     | 3.12.14 (uv-managed cpython, `.python-version`) |
| torch      | 2.15.0 (MPS built + available)            |
| transformers | 5.21.0 (has `Qwen3_5ForConditionalGeneration`) |
| gradio     | 6.31.0                                    |
| accelerate | 1.15.0                                    |
| safetensors| 0.8.0                                     |
| huggingface_hub | 2.3.0                                |
| tokenizers | 0.23.3                                    |
| numpy      | 2.5.4                                     |

## Load recipe (T3, Eng D9)

Rung 1 of the fallback ladder (Eng D2):

| piece             | value                                              |
|-------------------|----------------------------------------------------|
| model id          | `Qwen/Qwen3.5-4B` (public, ungated, apache-2.0)    |
| revision (at T3)  | `851bf6e806ef` (main; `verify_load.py` records it) |
| class             | `Qwen3_5ForConditionalGeneration`, in-transformers |
| trust_remote_code | `False` (no remote code needed)                    |
| dtype             | `torch.float16` (config native bf16 noted)         |
| device            | single MPS, eager (plain `.to("mps")`, no dispatch)|
| prompt wrap       | `tokenizer.apply_chat_template` (verified live)    |
| weights           | 9,319,737,856 param bytes, 2 safetensor shards     |
| disk budget       | 15 GiB (weights ~8.7 GiB + deps + cache headroom)  |
| memory measure    | `torch.mps.current_allocated_memory` (verified)    |

Code: `steer.load_tokenizer`, `steer.wrap_prompt`, `steer.load_model`.
Proof: `logs/load_proof.json` (written by `scripts/verify_load.py`).
Measured at T3 (M4 Pro, 24 GB): 4.54B params, ~9.4 GB peak MPS driver
memory after load, 16-token greedy probe in 3.0 s. First load took 334 s
including the one-time ~8.7 GiB download; weights are cached after that.
Note: transformers falls back to reference kernels for the linear-attention
ops (`causal_conv1d`, `flash-linear-attention` not installed — no macOS
wheels). Correct but slower; generation is still fast enough for v1.

Model notes for later tasks (observed at T3, not yet acted on):

- Multimodal architecture (`Qwen3_5ForConditionalGeneration`, text + vision).
  Text-only runs use the text path; T1 confirms hook placement on it.
- Text config: 32 layers, hidden 2560, intermediate 9216, hybrid
  linear/full attention. Layer/neuron bounds checks (T1/T5) use these.
- Weight map shows `model.language_model.layers[L].mlp.down_proj.weight`
  (T5 hardcodes the verified path plus an assert; D1).

## Fallback ladder (Eng D2 + D8, for T2)

Top first; stop at the first rung that passes. Presets bind to model id:
a fallback invalidates evil-qwen rows until re-derived.

1. qwen3.5-4B fp16 on MPS eager (test: loads + generates + hooks fire).
2. Same-family smaller, e.g. `Qwen/Qwen3.5-2B` (verified to exist at T3).
3. GGUF-in-transformers if hooks work (hook + layout + norm-scale check first).
4. CPU run (slow accepted).

Recovery: contrast-prompt activation diff on the new model + re-run the Day-0
gate, timeboxed half a day.

## Layout

- `steer.py` — engine (T3: load recipe only; hooks T1, ladder T2, calib T4).
- `app.py` — Gradio UI (T6, not built yet).
- `scripts/check_env.py` — env gate (T3).
- `scripts/verify_load.py` — load proof (T3).
- `presets/` — recipe presets (T1, not built yet).
- `tests/` — unit suite + integration lane (T8/T9, not built yet).
- `logs/` — local run receipts, not committed.

## Out of scope for v1

Filmstrip, recipe replay (v1 writes the string, v1.5 parses it), neuron
graph, public repo, CI. See design doc.
