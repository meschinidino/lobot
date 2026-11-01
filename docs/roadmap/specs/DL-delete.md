# DL Delete (run-time ablation)

Owner: backend agent. Lane: no-model for the code and tests; one
model slot for the side-effect + guard checks. Depends on: engine
ablation (exists, D13). Time: one day.

## Goal

Run-time ablation for eligible traits, with side-effect checks:
a Delete that removes a trait at run time, reports what else it
touches, and re-checks the guard with Delete active.

Acceptance (from `later.md`): Delete report cards with side
effects on prompts that elicit the trait; guard re-checked with
Delete active.

## Read first

- `src/model_lobotomy/engine.py`: `Ablation`, D13 order,
  `overlap_warning`.
- `src/model_lobotomy/server.py`: `/api/update` (trait filmstrip),
  `_trait_vector`, `_verified_guard`.
- `src/model_lobotomy/trait_eval.py`: `delete_gains` (removal
  measurement), `NEUTRAL_INSTRUCTION`.
- `src/model_lobotomy/guard.py`: `check_verified`.

## Deliverables

- `POST /api/delete` (NDJSON like `/api/update`): run one prompt
  with the trait's direction ablated (ablation at the knob layer)
  vs clean, streamed as two token columns (`doses` fixed to
  `[0.0, 1.0]` meaning off/on; the on-column carries
  `dial: {trait: value}` when the trait has dial anchors).
  Request: `{"prompt", "trait", "max_new_tokens", "guard",
  "research_mode"}`. Same guard-first shape as update (the guard
  checks the ablation, not a dose).
- `POST /api/delete/check {"trait", "prompts" | null}` (one JSON
  object, needs the model): the Delete report card with side
  effects. Trait side: removal gains on 12 trait-eliciting
  prompts (default: the trait's pos-prompted bank sample, fixed
  list in the code) with the shuffled-ablation control. Quality
  side: NLL delta on 12 neutral prompts (same sample
  unprompted) and the refusal-set pass rate with Delete active.
  Returns `{"card", "nll": {"base_mean", "deleted_mean",
  "delta"}, "refusal": {"pass_rate", "n"}, "guard"}` where guard
  is the `check_verified` verdict on the ablation.
- Eligibility: Delete serves traits with a finished knob and a
  saved direction; else 422 ("Delete is not eligible for ...":
  no knob). The check route MEASURES the card (pass or fail);
  eligibility never requires a passing card (`later.md` asks
  for report cards with side effects, not passing cards — the
  first draft required a passing card, which left the eligible
  set empty since no shipped delete card passes). `GET
  /api/traits` items gain `"delete_eligible": bool`.
- `tests/test_delete.py` (no model; scripted engine).
- `docs/reference/api.md` updated. Report `reports/DL.md`.

## Method notes

- Ablation = `Ablation(layer, direction)` at the knob layer
  (same site as Create's delete verb). Dose is binary (off/on);
  no dose tuning.
- Trait-eliciting prompts: 12 fixed bank prompts, pos-prefixed
  with the trait template (the trait shows) — removal gain =
  base judge minus deleted judge, like `delete_gains`. Control:
  shuffled-direction ablation on the same prompts.
- NLL side effect: mean NLL of the clean reply under the clean
  model vs mean NLL of the deleted reply under the clean model
  (same `spike.nll` as tune). A big positive delta means Delete
  mangles fluency; reported, no gate.
- Refusal side effect: refusal-judge pass rate (score >= 0.5) on
  the guard refusal set with Delete active, base vs deleted.
  Reported, no gate (the guard verdict is the gate).
- Guard: `check_verified` on the ablation edits. A blocked
  ablation is served as `blocked` (delete route) / reported
  (check route) and never streams.

## Tests (no model; scripted engine)

- `test_delete_streams_two_columns` (off/on texts differ).
- `test_delete_guard_block_first` (block ends the stream).
- `test_check_card_shape` (card + nll + refusal + guard keys).
- `test_ineligible_trait_422` (no knob).
- `test_traits_list_marks_eligibility`.
- `test_ablation_actually_applies` (edits carry Ablation).
- 409/503 follow the update route tests.

## Acceptance checks

```bash
uv run python -m unittest tests.test_delete -v
uv run python -m unittest discover -s tests -t .      # full suite passes
# live: POST /api/delete/check for pirate (eligible) prints a card;
#      the guard dict shows a fresh verified verdict on the ablation
```

Gate: the check route returns a measured card with NLL delta
and refusal rate, plus a verified guard verdict, for at least
one eligible trait. Reported numbers decide (pass or fail).

## Done means

All of the above, plus a report with the exact check output for
the eligible traits. `docs/reference/api.md` documents both routes and the
eligibility flag.

## Out of scope

Weight-space deletion; multi-trait ablation (CB covers pairs);
any UI (the UI lane adds the Delete button + check panel).

## Risks

- Measured delete cards fail (no shipped delete card passes).
  The check still reports them with side effects; a failing
  card is data, not a gate miss.
- Ablation at run time shifts the guard (blocks a useful
  Delete). Report the verdict; do not soften the guard.
