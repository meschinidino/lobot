# C2 Fast describe-to-patch

Owner: backend agent. Lane: no-model for the code and tests; long
model slots for the fast/full matrix. Depends on: Create (wave 1).
Time: three days (most of it model-lane runs).

Note: this 100X `C2` reuses the 10X `C2` label (hostility family,
reported in `reports/day-2-backend.md`). This spec is the 100X
workstream from `later.md`.

## Goal

Create in about 1 minute through three levers: fewer prompts,
cached captures, and early stop when the gate is clear. The full
schedule stays available and numerically unchanged.

Acceptance (from `later.md`): median Create time <= 2 min on
Llama; same pass/fail as the full schedule on the known families.
`later.md` says "8 known families"; the store holds 7 (blunt,
cheerful, hostility, melancholic, paranoid, pirate,
shakespeare). C2 validates blunt (the last unvalidated family)
and runs all 7; the "8" is reported as a table-vs-store gap, not
re-tuned around.

## Read first

- `src/model_lobotomy/trait_eval.py`: `run_trait` and every stage
  (this is the code C2 parametrizes; read all of it).
- `src/model_lobotomy/create_job.py`: `start` and `_run`.
- `src/model_lobotomy/evaluation.py`: `make_splits`,
  `report_card_pass`, `bootstrap_ci`.
- `scripts/run_trait.py`, `scripts/validate_judge.py`.

## Deliverables

- `trait_eval.run_trait(..., schedule="full")` with a `SCHEDULES`
  table (see Method notes). `schedule="full"` reproduces today's
  numbers exactly (same prompts, same order, same generations).
- Capture cache: `store/captures/<slug>/<trait>-seed<N>.npz`
  (extraction A/B per layer) keyed by bank version, model
  revision, pos/neg text and split contents. Both schedules use
  it. Committed: nothing (runtime cache, gitignored).
- Base-generation cache inside one run: greedy base replies are
  deterministic, so each prompt's base text generates once and is
  shared across verbs, controls and baseline instructions. Both
  schedules use it; numbers are unchanged, generations drop.
- Early-stop rule `should_stop_early(trait_gains, control_gains)`
  (pure function, unit-tested): after the minimum pairs, stop
  when the card passes AND the trait lower CI stays above 0.05;
  else extend to the maximum. Fast final verbs and fast baseline
  use it; full never stops early.
- `CreateRunner.start(..., schedule="full")`,
  `POST /api/create {"schedule": "fast"|"full"}` (default
  **full**; fast stays available but preliminary — amendment
  after the parity failure, see the report: a default that
  returns no_knob for known knobs misleads). Status and
  `report.json` record the schedule, per-stage seconds, and every
  early-stop decision. `scripts/run_trait.py` gains `--schedule`
  (default full: the script is the reference tool).
- Blunt validated (`validate_judge.py --trait blunt`, agreement
  >= 0.80) so all 7 families run.
- `tests/test_create_fast.py`: schedule plumbing, cache key
  round-trip and invalidation, early-stop rule cases, base-cache
  sharing counts on a fake engine, full-schedule prompt order
  unchanged. No model.
- `docs/reference/api.md` updated. Report `reports/C2.md`.

## Method notes

- Fast schedule (fixed before measuring):
  - generations: 24 tokens (full: 60);
  - tune: top 2 layers, doses [0.0, 1.0, 2.0], 5 tuning prompts,
    same NLL coherence rule (`best_dose` unchanged);
  - read gate: fit 20 (captures), heldout 16;
  - final verbs and controls: paired prompts from 6 up to 10
    with early stop;
  - baseline: 3 instructions x 5 tuning prompts for top-1, final
    6 up to 10 with early stop, neutral at the same n;
  - guard: unchanged (`check_verified`, 24-token confirm).
- Full schedule: every number as today. The only full-path
  changes are the two caches (numerically identical: same
  captures, same greedy texts).
- Progress callbacks keep working: totals are the maxima; early
  stop reports the done count.
- Verdict comparison is on `verdict` (knob/no_knob/blocked) at
  seed 0 per family. Guard runs record-only (no `--guard` flag)
  so `blocked` verdicts do not appear either side; the guard
  dicts are still compared for would_block agreement and
  reported.
- Reference full verdicts: the 3 recent seed-0 records
  (cheerful, pirate, shakespeare: all knob) stand only if
  `trait_eval` full-path behavior is unchanged (assert by
  re-running full for pirate and comparing verdict + frozen +
  report cards to the record). Else run full fresh for all 7.
  The other 4 families get fresh full runs either way.

## Tests (no model; fake engine)

- `test_schedule_defaults_fast_on_api_full_in_library`.
- `test_full_schedule_prompt_order_unchanged` (stage prompt lists
  equal today's, on a recording fake).
- `test_capture_cache_round_trip_and_key_mismatch` (pos text
  change invalidates).
- `test_base_cache_generates_once_per_prompt` (counts `run`
  calls across two verbs).
- `test_should_stop_early_*`: clear pass stops, borderline
  extends, fail extends, short lists never stop.
- `test_fast_uses_fewer_prompts` (counts per stage on the fake).
- `test_unknown_schedule_rejected` (API 422, library ValueError).

## Acceptance checks

```bash
uv run python -m unittest tests.test_create_fast -v  # all pass
uv run python -m unittest discover -s tests -t .     # full suite passes
uv run python scripts/validate_judge.py --trait blunt   # >= 0.80
# per family at seed 0 on Llama: fast via API or run_trait.py,
# full via run_trait.py --schedule full (fresh where needed)
```

Gate: median fast wall time <= 120 s over the 7 families AND all
7 fast verdicts equal their full verdicts. A miss on either is
reported as failed with the per-family table; the fast numbers
above are not re-tuned to pass.

## Done means

All of the above, plus a report with the per-family table
(family, fast seconds, fast verdict, full verdict, match), the
median, the capture-cache repeat-run speedup, and the blunt
validation numbers. `docs/reference/api.md` documents `schedule`.

## Out of scope

New families (C3); faster generation kernels; skipping the guard
confirm; any UI (the UI passes `schedule` through when ready).

## Risks

- The median lands above 2 min (judge forwards and guard confirm
  do not shrink). Report the miss with the stage breakdown.
- Short generations flip a verdict (judge on 24 vs 60 tokens).
  Report the mismatch; do not lengthen fast prompts to pass.
- Full reference runs take ~40 min each: ~3 h of lane for the 4
  missing families. Run them in background, one at a time.
