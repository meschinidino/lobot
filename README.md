# model-lobotomy

Visual neuron-steering workbench. Local-first, one model, before/after loop.
V1: load Qwen3.5-4B, pick layer/neuron/dose, run the same prompt with and
without steering, see the diff, copy a paste-ready artifact for X.

Design doc (source of truth, Status APPROVED):
`~/.gstack/projects/model-lobotomy/office-hours-design-20261028-233909.md`

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
2. Same-family smaller `Qwen/Qwen3.5-2B` (verified live: sha `15852e8c1636`,
   public ungated; T2 `scripts/check_engine_t2t4t5.py`).
3. GGUF-in-transformers if hooks work (hook + layout + norm-scale check first).
4. CPU run (slow accepted).

Recovery: contrast-prompt activation diff on the new model + re-run the Day-0
gate, timeboxed half a day.

## Layout

- `steer.py` — engine (load T3, hooks/runner T1, ladder T2, calib T4,
  path T5, cap + measurements + artifact template T7).
- `app.py` — Gradio UI (T6). Run: `uv run python app.py` → http://127.0.0.1:7860.
- `scripts/run_artifact.py` — headless T7 runner: one before/after pair →
  `artifacts/v1-<UTC stamp>/` (`artifact.txt` + `post.txt` + `run.json`).
  Human posts; nothing posts. Cap 256 tokens; per-side prefill/decode +
  MPS peak per run.
- `scripts/check_env.py` — env gate (T3).
- `scripts/verify_load.py` — load proof (T3).
- `scripts/check_engine_t2t4t5.py` — engine smoke (T2/T4/T5; `--with-model`).
- `presets/evil-qwen.json` — evil-qwen recipe (T1).
- `tests/` — 5-file unit suite (T8): `uv run python -m unittest
  discover -s tests -t .` — 66 tests, green in seconds, no weights.
  Integration lane (T9): `MODEL_LOBOTOMY_INTEGRATION=1 uv run python -m
  unittest tests.test_steering_loop -v` — 5 model-backed tests, PASS
  (see `T9_VERDICT.md`); skipped by default.
- `logs/` — local run receipts, not committed.

## Out of scope for v1

Filmstrip, recipe replay (v1 writes the string, v1.5 parses it), neuron
graph, public repo, CI. See design doc.
