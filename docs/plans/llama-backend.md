# llama.cpp/Metal backend plan

Parallel, flag-gated alternative to the PyTorch Engine. Torch stays
default. Select with `LOBOT_BACKEND=llama` (any other value raises).

Decisions: two Claude consults (binding). Session id in
`.context/claude-session-id`; resume it for further doubts.

## Shape (D1)

- Custom C++ daemon `native/lobot-llamad/`, stdio JSON-lines, one
  child process per `LlamaEngine`. Stdout is protocol only; llama
  logs go to stderr via `llama_log_set`. Python drains stderr on a
  thread.
- Ops: `info`, `generate {ids, max_new, stop_ids, cvec_b64,
  il_start, il_end}`, `prefill` (full last-position logits, b64
  f32 LE), `cancel` (stdin reader thread sets a flag; decode loop
  checks between tokens).
- Tokenization stays in Python: HF tokenizer wraps (`wrap_prompt_
  notk`), tokenizes, and cumulative-decodes exactly like torch.
  GGUF template/tokenizer never used. Greedy (argmax) decode.
- Each request: `llama_memory_clear`, set cvec or clear with
  `data=NULL`, decode. No cross-request state. No prompt cache v1.
- Daemon crash raises loudly; never silent-fallback to torch.
  Restart on next `load`.
- `load` checks daemon `n_vocab`/`n_layer`/`n_embd` against the
  torch vocab size and spec; mismatch stops.

## Edits (D2)

- VectorEdit: fully supported. NeuronEdit: supported via exact
  conversion (down_proj has no bias, so `d` at MLP input n equals
  `d * W_down[:,n]` on the residual; llama.cpp applies cvec at the
  same site, after the FFN residual add).
- Merge into one F32 cvec buffer, slot `L-1` per torch layer L,
  strengths baked into floats, scale 1.0, `il_start`/`il_end` =
  min/max touched layers.
- W_down columns come from HF safetensors at the pinned revision
  (safe_open, one layer at a time, cached per model+revision).
  Never from the GGUF (quant would change steering size).
- `steer.column_norm` refactors to a pure function on a weight
  tensor. Both backends share one `_neuron_deltas` math.
- Ablation, Read, dials, capture: whole EditSet raises
  `UnsupportedError` naming the torch backend. Never partial.
  Layer-0 edits raise (llama.cpp forbids direction.0).
- Medians come only from the torch calibration cache. Miss
  raises: calibrate on `LOBOT_BACKEND=torch` first.

## Toolchain + GGUFs (D3)

- `scripts/llamacpp.lock`: one line, the pinned sha.
- `scripts/build_llamacpp.sh`: clones into gitignored
  `third_party/llama.cpp` at the lock sha; builds libs,
  `llama-quantize`, and `lobot-llamad`. Metal on.
- Convert with `PYTHONPATH=third_party/llama.cpp/gguf-py` under
  `uv run` (matching gguf-py, not the env package).
- Convert the exact revision torch loads (same sha resolution as
  `Engine.load`). Qwen3.5: text model only, no mmproj.
- Layout: `models/<slug>/<rev12>-<quant>.gguf` + sidecar `.json`
  (HF repo, revision, llamacpp commit, quant, sha256).
  `fingerprint()` reads the sidecar. `models/` and
  `third_party/` go in `.gitignore`.
- Regenerate Llama F16 with the pinned toolchain; do not reuse
  `logs/gate_llamacpp/model-f16.gguf` (unknown provenance).
- F16 = parity lane. Q8_0 = speed lane. Q4_K_M = report-only.

## Wiring (D4 + conflicts)

- `src/model_lobotomy/llama_backend.py`: `LlamaEngine` with
  `backend = "llama"` (`Engine` gains `backend = "torch"`).
- `engine.get_engine()` reads `LOBOT_BACKEND` once on first call,
  lazily imports `llama_backend` for `llama`, caches. Never
  re-reads.
- `LlamaEngine.model` is a strict `LlamaModelHandle`: `__call__`
  answers only `[1, T]` tensor input via daemon prefill with cvec
  cleared, returning logits readable only at `[0, -1]`; every
  other attribute raises `UnsupportedError`. Loaded-checks pass
  untouched; judge/guard work through the handle.
- `app.load_engine`: `assert_text_dims` only when
  `_ENGINE.backend == "torch"`.
- `_calibrate`: catch `UnsupportedError`, re-raise with the
  calibrate-on-torch message.
- `FEATURES = ("dial", "capture", "reader", "sae", "create",
  "verify", "ablate")`. `Engine.supports()` all True;
  `LlamaEngine.supports()` all False.
- `_state()` gains `backend` + `torch_only` on both backends.
  Only torch-visible change.
- `_need_torch(eng, feature)` helper → HTTP 501, called before
  any `StreamingResponse`. Route disposition:
  - Work: `/api/run` (no dial), `/api/chat*` (no dials),
    drafts validate, install verify, drafts,
    combine/check + combine/installed (vector-only),
    update (sweep must confirm).
  - Gate: `/api/run`+dial → `dial`; `/api/neurons` →
    `capture`; neuron/find/lift → `reader`; feature* →
    `sae`; `/api/create` → `create`; `/api/verify*` →
    `verify` (stays closed until parity passes; ask first);
    `/api/delete` + `/api/delete/check` → `ablate`;
    combine routes → `ablate` when any patch has
    `meta.edit_kind == "ablate"`. `_delete_stream` stays
    byte-identical.
  - Backstop: `@api.exception_handler(UnsupportedError)` →
    501. Never replaces explicit gates; cannot help
    mid-stream.
- `run` returns all torch keys; `peak_mib`/`driver_mib` None;
  `hook_calls` = touched layers × forwards, 0 when empty;
  `reads={}`. `run_batched*`: sequential singles,
  `fallback_used=False`. `prefill` returns 1-D f32 torch tensor.
  `fingerprint()` = torch keys + backend/gguf_sha256/quant/
  llamacpp_commit. Median lookup keeps the torch key.
- UI gating for torch_only controls: visual change, needs
  DESIGN.md review. Ask before building.

## Tests (D5)

- Unit (fake daemon, label as such): cvec layout, layer-0
  rejection, whole-EditSet ablation rejection, hook_calls,
  cancel, crash, result-key equality; plus the torch-only
  conversion test (neuron pre-hook vs F32 residual add:
  logits cosine ≥ 0.9999, same 32 greedy tokens).
- F16 parity per prompt × {baseline, vector, neuron,
  vector+neuron}: 32 greedy tokens match, else first split at
  index ≥ 16 with torch logit gap < 0.5; prefill cosine ≥
  0.999, same top-1, refusal-starter |Δ| ≤ 0.01. Llama first;
  Qwen counts only after its own pass.
- Q8_0: baseline prefill top-1 ≥ 90%, refusal |Δ| ≤ 0.03,
  identical guard verdicts. Q4_K_M report-only unless it meets
  the Q8 bar.
- Speed: warm model, median of 5, TTFT + tok/s + EditSet-switch
  cost; torch fp16 MPS vs llama F16 vs llama Q8_0; daemon
  timings + Python wall-clock, both reported.
- Guard: 3 EditSets incl. one torch-blocked and one near
  threshold; identical verdicts on F16.
- Route sweep on `LOBOT_BACKEND=llama` + fake daemon: every
  `/api/*` once; 2xx or 501 only. Full torch suite unchanged
  except full-`_state()` comparisons.

## Results (2026-10-31)

- F16 parity PASS on both models: 12/12 generations byte-exact,
  prefill cosine >= 0.999998, guard verdicts identical (3/3).
- Q8_0 meets its bar on both: baseline top-1 100%, refusal
  |d| <= 0.006, verdicts identical, 9-10/12 exact (near-tie
  splits only). Reports: `logs/parity_llama/`.
- Speed (median of 5, 32 tokens): llama Q8_0 beats torch
  (Llama 40.1 vs 28.5 tok/s; Qwen 41.2 vs 12.7); F16 about
  tied with 3-6x faster TTFT. cvec build 0.4 ms.
- Route sweep `tests/test_llama_routes.py`: 16/16, every
  route 2xx/501/422-by-design, no 500. Full suite 569 green.
- Fixes found by parity: daemon emits the stop token (HF
  parity), stop ids come from the text config (Qwen gen eos
  248044 != tokenizer eos 248046), vocab check reads the
  config size (Qwen pads to 248320).

## Build order

1. Lock + build script + daemon source; build (long).
2. `steer.column_norm` refactor + weight-column reader.
3. `llama_backend.py` + `get_engine` switch + `app`/`server`
   gates + `_state` keys.
4. Unit tests + conversion test (no weights).
5. Convert Llama F16 + Q8_0 (book model lane; conversion loads
   full weights).
6. Parity + speed + guard scripts; run on Llama, then Qwen.
7. Route sweep. Report with labels (fake vs end-to-end).
