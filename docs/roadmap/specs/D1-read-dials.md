# D1 Read dials in /api/run

Owner: backend agent. Lane: no-model (one optional model check at the end).
Depends on: contracts section 3. Time: half a day.

## Goal

Let `/api/run` return, for each generated token, how strongly a saved trait is
present in the model at that moment, on the scale 0 = neutral mean,
1 = trait mean. This is the "see it" part of the loop, and it feeds the read
dials in the UI (U3).

## Read first

- `engine.py`: `Read`, `EditSet`, `Engine.run()` (it collects `reads`),
  `Engine.run_stream()`.
- `trait_eval.py`: where the frozen direction and the contrast-set statistics
  are computed; find or add the neutral and trait means of the projection.
- `create_job.py` and a sample `jobs/<id>/trait/report.json` layout.
- `server.py` `_run_stream`.

## Deliverables

- `traits.py` helper (or a function in `trait_store.py`):
  `load_dial(trait_name, model_id) -> Dial(layer, unit, mu_neutral, mu_trait)`
  from the newest done job with verdict knob for that family and model.
- `/api/run` accepts `"read": [names]`; validates before generation (404 with
  "No saved trait named X for this model."); adds `Read(layer)` edits to both
  the baseline and the steered pass; projects per token; adds `reads` to the
  result exactly as contracts section 3.
- If the report lacks `mu_neutral` and `mu_trait`, compute them once from the
  trait's contrast prompts at load time and cache them in
  `jobs/<id>/trait/dial.json`.
- `docs/api.md` update.
- Tests.

## Method

projection_t = hidden_t · unit (layer output after all edits, D13 order)
value_t = (projection_t - mu_neutral) / (mu_trait - mu_neutral)

Per generated token: use the hidden state at the position that produced the
token (prefill last position for token 0, then each decode step). Make sure
`Read` collects every decode step, not only the last forward pass; if
`Engine.run` keeps only the last tensor, extend it to append per step (and test
it).

## Tests (fake engine with a planted direction)

- `test_read_values_scale`: hidden states equal to mu_neutral·unit give 0,
  mu_trait·unit give 1.
- `test_reads_one_value_per_generated_token` for both passes.
- `test_unknown_trait_404_before_generation` (generation mock never called).
- `test_read_does_not_change_output_ids`: same token ids with and without `read`.
- `test_reads_absent_when_not_requested` (result unchanged for old clients).
- `test_dial_json_cached_once`.

## Acceptance checks

```bash
uv run python -m unittest discover -s tests -t .   # all pass
```
Optional model check: with a saved trait on Llama, `/api/run` with
`"read": ["<trait>"]` returns per-token values, and the steered pass shows a
higher mean than the baseline. Paste the two means in the report.

## Done means

Route change, helper, docs, tests, report.

## Out of scope

Streaming dials token by token (later; the UI animates from the final list).
Chat mode.
