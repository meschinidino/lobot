# F1 Clean features (sparse autoencoder)

Owner: backend agent. Lane: no-model for the code and tests; one model
slot to cache activations, train, and measure. Depends on: R1, R3.
Time: two days.

## Goal

Train a sparse autoencoder (SAE) on chosen Llama layers on the Mac,
then read and steer SAE features instead of single neurons:

- **Read**: a feature card (what the feature pushes toward, what it
  fires on) through the same reader math as neurons.
- **Steer**: push a feature direction and measure lift on its top
  words, through the same vector-edit path as trait knobs.

Acceptance (from `later.md`): on 50 random features, reader
coherence median >= 0.6 (reported next to the single-neuron median
on the same layers); steering a feature has lift >= 3 for its top
words on at least half the features.

## Read first

- `src/model_lobotomy/reader.py`: `pushes_toward`, `coherence`,
  `fires_on`, `steer_lift`, `_projected_logits`.
- `src/model_lobotomy/engine.py`: `VectorEdit`, `capture`, `run`.
- `src/model_lobotomy/steer.py`: `calibrate_layer_medians`.
- `scripts/build_reader_cache.py`: how the neuron cache is built.
- `tests/test_reader.py`: how the fake models are built.

## Deliverables

- `src/model_lobotomy/sae.py`: TopK SAE (train, save, load),
  activation caching, feature fires cache, feature cards.
- `src/model_lobotomy/reader.py`: two generalizations (see Method
  notes). No behavior change for existing callers.
- `scripts/train_sae.py --model <id> --layers 12,20 ...`: cache
  activations, train one SAE per layer, write the store dir.
- `scripts/eval_features.py --model <id> --layers 12,20`: the
  acceptance measurement (features and the neuron baseline).
- Routes: `GET /api/features`, `GET /api/feature/{layer}/{index}`,
  `POST /api/feature-lift` (see Contracts addition below).
- `tests/test_sae.py` (no model).
- `docs/reference/api.md` updated. Report `reports/F1.md`.

## Method notes

- **SAE**: TopK SAE on the residual stream (`block_out`) at each
  chosen layer. Hidden width H, F = 8 * H features, top-k = 32
  active per token. Loss is plain MSE; decoder rows are
  re-normalized to unit norm after each step. Dead features are
  counted (never active on the sample) and reported, not revived.
- **Data**: `data/reader_sample.txt`, same sample as the neuron
  cache. Cache per-token `block_out` vectors (skip BOS per
  document, like R1). Train up to 8 epochs, batch 4096, Adam
  lr 3e-4, seed 0. All values are flags on `train_sae.py`; the
  defaults above are the shipped setting.
- **Layers**: 12 and 20 on Llama 3.2 3B (mid and late; the demo
  neuron sits at 19). One SAE per layer.
- **Read**: a feature direction is decoder row i (unit norm).
  `reader.pushes_of_direction(eng, direction, k)` projects any
  hidden vector through gamma and the head; `pushes_toward` calls
  it (same numbers as today). Coherence is `reader.coherence` on
  the up tokens. Feature fires come from encoder activations over
  the cached sample, stored like `fires_on.npz` (`act [F, k]`,
  `pos [F, k]`).
- **Steer**: a feature steers as a `VectorEdit` with the decoder
  direction, scale = the layer median norm, dose = 1.0 (the same
  dose unit as Create knobs). `reader.steer_lift_vector` is the
  vector version of `steer_lift` (same prompts, same math).
- **Acceptance detail** (fixed before measuring): 50 features per
  layer are drawn with seed 7 from live features (active at least
  once on the sample). Top words = the top-3 whole-word up tokens
  (fallback: top-1 up token as is). Coherence is on up-12.
  Lift is measured at dose 1.0 with the 8 fixed lift prompts.
  The neuron baseline is 50 random neurons per layer (seed 8).
  Gate: feature coherence median >= 0.6 AND lift >= 3 on >= half.
  Doses 0.5 and 2.0 are recorded as diagnostics only.
- **Store layout** `store/sae/<model-slug>/L{layer}/`:
  `sae.safetensors` (encoder, decoder, biases), `config.json`
  (layer, hidden, features, top_k, seed, sample tokens, train
  stats), `fires.npz`, `tokens.json`. Weights are NOT committed
  (see `.gitignore`); `config.json` files are committed so a
  checkout names what to train.
- **No-model rule**: training and eval scripts need weights and
  the model lane. Tests use a tiny fake model and random acts.

## Contracts addition (100X section in contracts.md)

- `GET /api/features` -> `{"saes": [{"model_id", "layer",
  "features", "top_k", "live", "dead", "mse"}]}`. 200 with an
  empty list when nothing is trained. No model needed.
- `GET /api/feature/{layer}/{index}?k=12` -> `{"model_id",
  "layer", "index", "live", "pushes_toward": {up, down,
  "coherence", "layer_frac"}, "fires_on": [...] | null,
  "fires_on_cache": {"built", "tokens"}}`. 400 when the index is
  out of bounds, 404 when no SAE is trained for the layer, 503
  without weights or in replay mode.
- `POST /api/feature-lift` body `{"layer", "index", "dose",
  "words" | null, "prompts": null}` -> the `steer_lift` dict.
  `words: null` uses the feature's top words (same rule as the
  acceptance). Errors like `/api/lift`, plus 404 without an SAE.

## Tests (no model; tiny fake model + random acts)

- `test_topk_forward_shapes_and_sparsity`: codes have exactly k
  nonzeros per row; recon has the input shape.
- `test_train_loss_falls`: 30 steps on random acts lower MSE.
- `test_decoder_rows_unit_norm_after_train`.
- `test_save_load_round_trip`: equal codes on fixed input.
- `test_dead_count_marks_silent_features`.
- `test_cache_round_trip`: cache acts on 3 tiny documents, top
  contexts and marks point at the right token.
- `test_pushes_of_direction_matches_manual_projection` and
  `test_pushes_toward_unchanged` (same numbers as before).
- `test_lift_vector_equals_one_at_zero_dose` and
  `test_lift_vector_above_one_for_planted_direction`.
- `test_feature_routes_*`: list empty, card 404 without SAE,
  card and lift against a tiny trained SAE (fake engine), 503
  with NO_MODEL, hook count zero after each call.

## Acceptance checks

```bash
uv run python -m unittest tests.test_sae -v        # all pass
uv run python -m unittest discover -s tests -t .   # full suite passes
uv run python scripts/train_sae.py --model meta-llama/Llama-3.2-3B-Instruct --layers 12,20
uv run python scripts/eval_features.py --model meta-llama/Llama-3.2-3B-Instruct --layers 12,20
```

The eval prints per-layer: feature coherence median, neuron
coherence median, share of features with lift >= 3, dead share.
Both layers must pass the gate. A gate that fails is reported as
failed, with numbers; the dose and seed above are not re-tuned to
pass.

## Done means

All of the above, plus a report with the exact eval output, train
time and peak memory, and three example feature cards pasted as
text. `docs/reference/api.md` documents the three routes.

## Out of scope

Transcoders (MLP in-out mapping); cross-layer features; any UI;
using features inside Create (a later workstream may propose it).

## Risks

- Coherence median < 0.6: residual features may read worse than
  hoped. Report the miss; do not lower the bar.
- Lift < 3 on most features at dose 1.0: decoder directions may
  need a bigger dose. The 0.5/2.0 diagnostics show it; the gate
  stays at 1.0.
- Train time on the Mac CPU/MPS: batch small if memory complains;
  report time and peak memory either way.
