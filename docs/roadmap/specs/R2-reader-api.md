# R2 Reader API

Owner: backend agent. Lane: no-model. Depends on: contracts only (R1 for the
real functions; stub them until R1 merges). Time: half a day.

## Status

Click-a-word on the **prompt** is already built (wave 2): `POST /api/neurons`.
Do not rebuild it. R2 extends it to reply words and adds three routes.

## Goal

Expose the reader to the UI: click a word in either answer, neuron card,
find, lift. Exact shapes: `contracts.md` section 2.

## Read first

- `src/model_lobotomy/server.py`: how `/api/run` takes the Engine lock, checks `NO_MODEL` and
  `REPLAY`, and returns errors.
- `docs/api.md`: style for documenting routes.
- `tests/test_server.py`: `ServerStreamTest` (fake engine) and
  `ReplayModeTest`.

## Deliverables

- `POST /api/neurons` extension: optional `reply` (appended after the wrapped
  prompt; `token_index` may address it; response adds `reply_start`) and
  `pushes_toward_top` on each neuron (null until R1 merges).
- New routes in `src/model_lobotomy/server.py`:
  `GET /api/neuron/{layer}/{neuron}`, `GET /api/find`, `POST /api/lift`.
- Replay support for `GET /api/neuron/...`: serve from
  `store/reader/<model-slug>/cards.json` when present; otherwise 503 with
  "Neuron cards need the model, or a recorded cards.json for replay mode."
- `docs/api.md`: a section per route.
- Tests in `tests/test_server.py` (class `ReaderApiTest`).

## Steps

1. Add request models with bounds from the active spec (layer, neuron, k
   1..50, token_index >= 0, dose within DOSE_MIN..DOSE_MAX).
2. Each model-backed route: 503 in `NO_MODEL`; 409 if the Engine lock is held;
   take the lock, load the active model if cold (same as `/api/run`), call the
   `reader` function, release in `finally`.
3. `layers=8-27` parsing: inclusive range, validated against the spec.
4. `/api/neurons` and `/api/run` both calibrate medians the same way. Factor
   that into one helper and use it in both places (no behavior change).
5. `reply`: tokenize wrapped prompt + reply as one sequence, so token ids
   match what generation produced. Keep the BOS rule (index 0 has no row).
   Accept `reply_char` (contracts section 2) and map it to a token index
   with the tokenizer's offsets; 400 when it is outside `reply`.
6. Wire `pushes_toward_top` (top 3 up tokens) into each `/api/neurons` result.

## Tests (fake engine and a patched `reader` module)

- `test_neuron_card_shape` and `test_neuron_out_of_bounds_400`.
- `test_neuron_card_without_cache_has_null_fires_on`.
- `test_find_parses_layer_range` and `test_find_bad_range_400`, `test_find_empty_word_400`.
- `test_neurons_reply_tokens_addressable`, `test_neurons_reply_char_maps_to_token` and `test_neurons_without_reply_unchanged`.
- `test_neurons_pushes_toward_top_null_without_reader`.
- `test_lift_shape` and `test_lift_dose_out_of_range_422`.
- `test_model_routes_503_without_weights` (NO_MODEL), `test_model_routes_409_when_busy`.
- `test_neuron_card_replay_from_cards_json` and `test_neuron_card_replay_without_cards_503`.
- `test_reader_routes_reject_cross_origin_post` (Origin check covers the POSTs).

## Acceptance checks

```bash
uv run python -m unittest tests.test_server -v     # ReaderApiTest all pass
uv run python -m unittest discover -s tests -t .   # full suite passes
```
Model slot (optional, or R3 does it): start the server and run
```bash
curl -s localhost:7860/api/neuron/11/1343 | python -m json.tool | head -30
curl -s "localhost:7860/api/find?word=pirate&layers=14-27&k=5" | python -m json.tool
```
Both return the contract shapes with real values.

## Done means

Routes, docs, tests, report with the curl outputs (or "pending model slot").

## Out of scope

The reader math (R1). UI (U2). Caching cards for all neurons.
