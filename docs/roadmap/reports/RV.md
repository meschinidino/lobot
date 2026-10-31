# RV report: re-verify anywhere

Spec: `docs/roadmap/specs/RV-reverify.md`. Branch:
`feature/RV-reverify`, merged into `develop`.

## What changed

- `src/model_lobotomy/reverify.py`: the check as a library.
  `run_verify_patch(patch, *, quick, progress, cancel,
  engine)` returns the same `(code, report)` as the script
  (shipped mean/CI, measured mean, n, pass). Accepts a path
  or raw bytes; an injected engine is used as-is, else the
  patch model loads and unloads. The check itself is
  unchanged (same prompts, seed, judge, interval rule).
- `scripts/verify_patch.py` now calls the library. Same
  output, same exit codes. Existing script tests pass
  unmodified.
- Routes: `POST /api/verify {"trait_id", "quick"}` streams
  NDJSON progress per prompt pair then one `done` with the
  report; `POST /api/verify/upload` (multipart `.lobot`, at
  most 64 MB) runs the same quick check on an uploaded
  file. The patch model must equal the active model (409
  with the sentence). Non-zip and pickled uploads are
  rejected (422) without unpickling. `/api/stop` ends a
  run after the current prompt pair.
- CI: `docs/ci/verify-patch.yml` (reusable workflow, inputs
  `patch_path` + `quick`, fails on FAIL; staged because the
  push token lacks the `workflow` scope — a maintainer copies
  it to `.github/workflows/`)
  plus `docs/verify-ci.md` (one command, weights for Qwen
  and gated Llama, CI snippet, UI flow).
- `docs/api.md` documents both routes.

## Tests

- New `tests/test_reverify.py`: 13 tests, no model
  (library report shape, quick/full counts, bytes input,
  garbage rejection, route stream, 422/404/409, upload
  round trip, non-patch and oversize rejection, workflow
  YAML parses with the inputs).
- Full suite: 475 pass, 5 skipped (pre-existing skips).

## Acceptance (live, pirate knob, Llama-3.2-3B-Instruct)

- `POST /api/verify` quick: 16 progress events, then
  `done`: measured 0.745 inside shipped [0.728, 0.812],
  **PASS** (margin 0.017 above the lower edge).
- `POST /api/verify/upload` of the same file: identical
  numbers, **PASS**.
- `scripts/verify_patch.py knob.lobot --quick` in a fresh
  process (the CI steps): measured 0.745, **PASS**.
- Gate: **PASS**. A truly external machine is out of
  reach; the fresh-process script run is the recorded
  third-party proxy, and the report says so.

## Gaps

- Open-family patches (e.g. noir) fail the check with
  "unknown trait": the frozen judge table covers legacy
  families only. Same as before RV; RV did not change it.
- No GPU CI runners; CPU runs work and take longer.

## UI needs (Claude)

- Verify button per trait (`POST /api/verify`) with the
  progress bar (`done`/`total`) and the PASS/FAIL report.
- Upload flow against `POST /api/verify/upload` with the
  64 MB cap and the 422/409/413 messages.

Relaunch: `uv run python -m model_lobotomy.server`.
