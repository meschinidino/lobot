# R1 Reader core library

Owner: backend agent. Lane: no-model for the code and tests; one short model
slot to build the real cache (or hand that to R3). Depends on: nothing.
Time: one day.

## Goal

A small library, `src/model_lobotomy/reader.py`, that says what a neuron is about, two ways, plus
the tools to find neurons and prove a reading:

- **Pushes toward** (output side): the words a neuron makes more likely.
- **Fires on** (input side): the contexts that activate it most.
- **Find**: the neurons that push a given word.
- **Word neurons**: the neurons most active at one word of an answer.
- **Lift**: how much steering a neuron raises its predicted words.

Exact signatures and return shapes: `contracts.md` section 1.

## Read first

- `src/model_lobotomy/engine.py`: `Engine.clean()`, `Engine.capture()`, `Engine.run()`,
  `EditSet`, `NeuronEdit`, `_lm_head()`.
- `src/model_lobotomy/steer.py`: `resolve_down_proj`, `text_config`, `calibrate_layer_medians`.
- `tests/test_engine.py`: how the fake models are built.

## Deliverables

- `src/model_lobotomy/reader.py` with every function in contracts section 1.
- `data/reader_sample.txt` (at least 50,000 tokens) and `data/README.md`
  (sources, licenses, how it was assembled).
- `store/reader/lift_prompts.json` (8 prompts, see contracts section 5).
- `scripts/build_reader_cache.py --model <id>`: builds
  `store/reader/<model-slug>/` with `build_fires_on_cache`.
- `tests/test_reader.py`.

## Method notes

- **Pushes toward.** Use `W_down[:, n]` (shape `[hidden]`), multiply
  elementwise by the final norm weight (RMSNorm gamma; do not re-normalize a
  single vector), then `lm_head`. Llama 3.2 ties embeddings; use the module
  `_lm_head()` returns. Report `up` (largest) and `down` (smallest).
- **Fires on.** One forward pass per document, no edits, inside `eng.clean()`,
  capture point `mlp_in` (down_proj input). Keep a running top-k per neuron
  with `torch.topk` on `[L, N, k + T]` merges; do not store all activations.
  Skip the first token of each document (BOS).
- **Find.** For a word's first token id `t`: `scores[L] = (gamma * W_U[t]) @
  W_down[L]` gives one score per neuron in a layer; z-score within the layer.
- **Word neurons.** Regenerate greedily with the run's edits until
  `token_index`, then one forward pass capturing `mlp_in` at that position.
  z = (activation - layer mean) / layer std, using statistics from the same
  forward pass over the prompt. Return the top k by z.
- **Lift.** For each prompt, softmax of the next-token logits; read p of each
  word's first token; repeat with a `NeuronEdit` at `dose`. Lift = geometric
  mean over words and prompts of steered_p / base_p.
- Everything runs on CPU in tests and on `steer.DEVICE` for real.

## Tests (no model; use a tiny fake model with a real lm_head)

- `test_pushes_toward_matches_manual_projection`: equal to the hand
  computation on a tiny model.
- `test_up_and_down_sorted_and_k`: lengths and order.
- `test_coherence_examples`: `["pirate", "ship", "sea"]` → 1.0;
  `["▁ing", "##s", "x"]` → 0.0; stems counted once.
- `test_fires_on_cache_round_trip`: build on 3 tiny documents, the known
  highest activation is first, contexts and marks point at the right token.
- `test_fires_on_skips_bos`.
- `test_find_neurons_ranks_planted_neuron_first`: plant a neuron whose
  `W_down` column maps to a chosen token; it ranks first.
- `test_word_neurons_returns_planted_neuron`: plant a neuron that fires only
  on one token; it tops the list at that index.
- `test_steer_lift_above_one_for_planted_neuron` and `..._equals_one_at_zero_dose`.
- `test_reader_leaves_no_hooks`: `Engine.hook_count() == 0` after each call,
  including after an exception.

## Acceptance checks

```bash
uv run python -m unittest tests.test_reader -v        # all pass
uv run python -m unittest discover -s tests -t .      # full suite passes
python -c "import json;p=json.load(open('store/reader/lift_prompts.json'));assert len(p)==8"
```
Model slot (or hand to R3): `uv run python scripts/build_reader_cache.py --model
meta-llama/Llama-3.2-3B-Instruct` finishes, prints tokens processed and
seconds, and writes both files. Report time and peak memory.

## Done means

All of the above, plus a report with the cache build time on Llama, and three
example neuron cards (layer, neuron, up 12, fires_on 3) pasted as text.

## Out of scope

Sparse autoencoders or transcoders (see `later.md`). Any UI. Any route (R2).

## Risks

- The sample text is too narrow → neurons look junky. Mix fiction, encyclopedic
  text, dialogue and instructions.
- Memory spikes during the cache build → process one document at a time and
  merge top-k per layer; never keep `[T, L, N]` for more than one document.
