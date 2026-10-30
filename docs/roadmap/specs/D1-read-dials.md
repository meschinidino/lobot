# D1 Read dials: baseline values and per-token lists

Owner: backend agent. Lane: no-model (one optional model check at the end).
Depends on: contracts section 3. Time: 2 to 3 hours.

## Status

The streaming dial is **already built** (wave 2): `/api/run` with
`"dial_trait"` streams `dial: {trait: value}` on each steered `token` event and
puts the anchors in `complete.result.dial`. Do not rebuild it. This spec adds
only what the tree (U3) needs to compare the two answers.

## Goal

`complete.result.dial` carries one value per generated token for **both**
passes, so the UI can draw the baseline and steered dials side by side and
show "the steered answer is N× more <trait>".

## Read first

- `docs/api.md`, section `/api/run`, "Read dial".
- `server.py` `_run_stream` (search `dial_trait`, `dial.json`).
- `engine.py`: `Read`, `EditSet`, `run_stream`.
- `tests/test_server.py`: the existing dial tests (search `dial`).

## Deliverables

- The baseline pass gets the same `Read` edit as the steered pass. It must not
  change the baseline's tokens.
- `complete.result.dial` adds `"baseline": [...]` and `"steered": [...]`:
  one `{"text", "value"}` per generated token (value float or null, same
  scale as the streamed values; texts join to the pass's text).
  The streamed `steered` list and the final list are equal.
- Replay: when a recording has `dial`, replay returns it; else `dial` is null.
  `scripts/record_sweep.py --dial-trait <family>` records it (skip the flag
  when no Create job exists; print why).
- `docs/api.md` update. Tests. Report.

## Tests (fake engine with a planted direction)

- `test_dial_lists_one_value_per_token_both_passes`.
- `test_dial_baseline_tokens_unchanged` (same ids with and without `dial_trait`).
- `test_dial_streamed_equals_final_steered`.
- `test_dial_absent_when_not_requested` (result unchanged for old clients).
- `test_replay_returns_recorded_dial` and `test_replay_dial_null_when_not_recorded`.

## Acceptance checks

```bash
uv run python -m unittest discover -s tests -t .   # all pass
```
Optional model check: with a saved trait on Llama, run with `dial_trait` and
paste the baseline and steered means in the report. The steered mean must be
higher when the dose pushes toward the trait; if it is not, report the numbers
as they are.

## Done means

Route change, recorder flag, docs, tests, report.

## Out of scope

Several traits per run. Chat dials (later, CH).
