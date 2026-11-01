# DL report: run-time Delete (ablation)

Spec: `docs/roadmap/specs/DL-delete.md`. Branch:
`feature/DL-delete`, merged into `develop`.

## What changed

- `POST /api/delete {"prompt", "trait"}` streams one prompt
  clean (off) vs trait-ablated (on) as two columns. Stages:
  `weights`, `guard`, `columns`, `token` (`col` 0/1), `done`.
  The on-column carries dial values when the trait resolves
  a dial spec. Guard blocks end the stream first.
- `POST /api/delete/check {"trait", "prompts?"}` returns the
  ablation report card: removal gains with the shuffled
  control, NLL delta, refusal pass rate with Delete active
  vs clean, and a fresh verified guard verdict on the
  ablation. Default prompts are the first 12 of the bank.
- `GET /api/traits` items carry `delete_eligible` (finished
  Create job with a saved direction). Both delete routes
  return 422 for ineligible traits.
- `docs/reference/api.md` documents both routes.

## Tests

- New `tests/test_delete.py`: 10 tests (two-column stream,
  ablation actually applies, guard-block first, check card
  shape, prompt validation, ineligible 422, 503/409, card
  counts, bank default).
- Full suite: 438 pass, 5 skipped (pre-existing skips).

## Acceptance (live, Llama-3.2-3B-Instruct, pirate, L17)

- `POST /api/delete/check {"trait": "pirate"}`: card
  **passed** (trait mean 0.360, CI [0.246, 0.460], all four
  rules true), NLL delta +0.035 (0.199 to 0.234), refusal
  7/8 base and 7/8 deleted, guard verified not blocked
  (min ratio 0.963). 205.7 s. Gate: **PASS**.
- `POST /api/delete` streams both columns; guard verdict
  verified on the ablation before the first token.

## Gaps

- One eligible trait measured live (pirate). Other traits
  (cheerful, shakespeare, noir) are eligible but unmeasured.
- Delete is run-time only; it writes no weights and no
  patch file.

## UI needs (Claude)

- Two-column off/on view for `POST /api/delete` with per-
  token append by `col` and dial readout on the on-column.
- Check panel for the card: gains, NLL delta, refusal
  rates, guard verdict.
- Eligibility badge from `delete_eligible` in the traits
  list; 422 message for ineligible traits.

Relaunch: `uv run python -m model_lobotomy.server`.
