# FINAL REPORT — model-lobotomy v1 build (T1–T10)

Date: 2026-10-29. Machine: Apple M4 Pro, 24 GB RAM. Repo: local-only,
no remote, nothing left the machine. Model: Qwen/Qwen3.5-4B @
851bf6e806ef (public, ungated), rung 1 fp16 on MPS eager.

## Status per task

| task | verdict | evidence |
|---|---|---|
| T3 env + load recipe | DONE | `scripts/check_env.py` exit 0; `logs/load_proof.json`; commit `f642d5f` |
| T1 Day-0 spike gate | PASS | `T1_GATE.md`; `logs/spike_t1/`; commit `143ac75`; T10: `GATE: PASS (lane=add, wall=190.7s)`; post-fix re-run: `GATE: PASS (lane=add, wall=189.5s)`, C4 2/3 |
| T2 ladder + recovery | DONE | `steer.py` ladder; `scripts/check_engine_t2t4t5.py` 17/17 (24/24 `--with-model`); commit `26031f7`; post-fix `--with-model` re-run: 24/24 |
| T4 calibration | DONE | guard + tap + isolation + pooled median (`t1-v2`); `calibration.json` L13 9.19 / L14 10.63; same smoke; commit `26031f7` + pooling-fix commit |
| T5 path assert | DONE | hardcoded `model.model.language_model.layers[L].mlp.down_proj` + assert; same smoke; commit `26031f7` |
| T6 Gradio loop | DONE | `app.py`; real-engine smoke transcript; screenshots `logs/t6_ui_*.png`; commit `5d606fe`; T10 probe: HTTP 200 on 127.0.0.1:7860 |
| T7 cap + artifact | DONE | cap 256; `scripts/run_artifact.py` exit 0, post 279 chars; `artifacts/v1-*/`; commit `129d9a9`; re-run in T10: exit 0 |
| T8 unit suite | DONE (0 xfail) | 66 tests, no weights; commit `8459ecd`; pooling pin fixed: 71-test discover `OK (skipped=5)` |
| T9 integration lane | PASS | 5/5 model-backed, `T9_VERDICT.md`; commit `52f8683`; T10: `Ran 5 tests in 20.299s OK`; post-fix: `Ran 5 tests in 17.070s OK` |
| T10 grammar + docs + report | DONE | `presets/README.md`, README run commands, this file |

## What works

- Prompt → tweak → diff loop: baseline and steered columns, first-divergence
  highlight, bounded dose slider (-3..+3) + numeric fallback per row.
- Evil-qwen preset flips 2/3 eval prompts at threshold (additive lane,
  x1.0, pooled calibration); dose-0 and baseline-vs-baseline controls
  green; hooks fire every steered run.
- Every run exports a paste-ready X artifact: `artifact.txt` (full recipe
  string) + `post.txt` (279 chars, <= 280 asserted) + `run.json` (metrics).
  The human posts; nothing posts.
- Recipe string grammar fixed and documented (`presets/README.md`); v1
  writes conform (field-checked by `tests/test_artifact.py` + live artifact).

## Commands (all re-run live in T10 unless noted)

```sh
uv run python scripts/check_env.py                                  # exit 0
uv run python scripts/spike_t1.py                                   # GATE: PASS, 190.7 s
uv run python app.py  # -> http://127.0.0.1:7860 (probed HTTP 200, stopped)
uv run python scripts/run_artifact.py                               # exit 0, post 279 chars
uv run python -m unittest discover -s tests -t .                    # 71 tests OK
MODEL_LOBOTOMY_INTEGRATION=1 uv run python -m unittest tests.test_steering_loop -v  # 5/5 OK
uv run python scripts/check_engine_t2t4t5.py                        # 17/17 PASS
```

Not re-run in T10: `scripts/verify_load.py` (one-time load proof, T3 receipt
`logs/load_proof.json` stands; weights unchanged since).

## Blockers and errors hit (all resolved except noted)

No hard blocker stopped any lane: no gated weights, no missing API, no spend,
no publish. Errors hit and fixed during the build:

1. T1: D1 assert fired on the first spike attempt — the guessed path
   `model.language_model...` was wrong; fixed to
   `model.model.language_model.layers[L].mlp.down_proj` and re-run (T1_GATE.md).
2. T3: transformers falls back to reference kernels for linear-attention ops
   (no macOS wheels for `causal_conv1d`/`flash-linear-attention`) — slower
   but correct; generation still ~23 tok/s.
3. T6: `gr.State` excluded from API input order broke the smoke; fixed with a
   hidden `gr.Number` for active-row count.
4. T7: prefill timing without MPS sync undercounted (0.04 s); fixed with
   `torch.mps.synchronize` (0.25 s / 0.12 s).
5. FIXED post-T10: the T8 calibration-pooling pin — `calibrate_layer_medians`
   pooled over the LAST prompt only. Fix accumulates norms across prompts
   (`pooled[L]`), bumps `CAL_CODE_VERSION` to `t1-v2`, regens
   `calibration.json` (L13 9.19, L14 10.63), removes the `expectedFailure`.
   Gate re-run: PASS (C4 2/3, was 3/3); unit + integration + smoke all green.
6. OPEN by design: rung-4 CPU path written but never executed (ladder stops at
   passing rung 1); rung-3 GGUF has no verified checkpoint (D8 verify-first).

## Own-recipe search on Qwen (negative result, 2026-10-29)

Contrast-prompt activation diff (`scripts/derive_llama_recipe.py`, sets of
8 refused vs 8 benign analogs) + single-row validation + pair round 2.
Round 1 (raw |diff|, late layers): winners degenerated into repetition
loops at 150 tokens. Round 2 (standardized diff, layers 8-26): best pair
L18N1066@-1 + L17N1256@-1 scored 2/2 at 100 tokens but loops at 150;
lower doses refuse or lecture. No coherent refusal flip found. The
misleading `presets/qwen-ours.json` was deleted, not shipped. Working
demo stays on evil-qwen. Receipts: `logs/derive_qwen*/` (git-ignored).

## Llama recipe (success, 2026-10-29, subagent)

Same derivation pipeline on Llama 3.2 3B Instruct found single-row
L11 N1343 @+1 (`presets/llama-refusal.json`): baseline refuses both
evals, steered engages coherently at 150 tokens (verified by two
independent runs each: subagent + owner). Held-out sabotage prompt
also flips. Calibration cache committed
(`calibration.meta-llama-llama-3.2-3b-instruct.json`). Note: Llama
steered outputs are tamer than evil-qwen (engaged-but-hedged, not
overtly evil). Demo runs ~6 s (3B is fast).

## Next steps

1. Human: review `artifacts/v1-20261029-042822/` (`artifact.txt` +
   `post.txt`, 279 chars), post one before/after to X.
3. v1.5 (out of v1 scope): recipe replay/parse, dose filmstrip, neuron graph,
   public repo + one-click setup.
