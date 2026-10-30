# R3 Reader validation and demo-neuron search

Owner: model agent. Lane: model (book it). Depends on: R1 merged.
Time: one day, most of it model time.

## Goal

Find, on Llama 3.2 3B, a few neurons that pass **both** checks, so beat 1
(read) and beat 2 (steer) use the same neuron honestly:

- **Readable**: the reader's words make sense (coherence >= 0.5, fires_on
  contexts agree with the up words).
- **Steerable**: pushing it changes the answer coherently and raises its
  predicted words (lift >= 3) without loops.

If no single neuron passes, fall back to a trait direction from Create and say
so in the report.

## Read first

- `src/model_lobotomy/reader.py` (R1), `src/model_lobotomy/engine.py`, `scripts/derive_llama_recipe.py`
  (`coherent()` loop check), `presets/llama-refusal.json`.

## Deliverables

- `scripts/search_demo_neurons.py --model <id> --layers 12-27 --top 200`.
- `store/reader/<model-slug>/` with the fires_on cache (if R1 did not build it)
  and `cards.json` for every shortlisted neuron (for replay mode).
- `logs/reader/<model-slug>/shortlist.json`: every candidate with its numbers.
- `docs/roadmap/reports/R3.md`: the top 5 neurons, each with up words,
  fires_on contexts, lift, the steered answer at the chosen dose, and a
  recommendation for the demo.

## Steps

1. Build the fires_on cache for Llama if missing (`build_reader_cache.py`).
2. Candidate pool: for layers 12-27, rank neurons by `pushes_toward` coherence;
   keep the top 200. Also add neurons from `find_neurons` for 10 demo-friendly
   words (pirate, ocean, music, food, space, weather, poetry, money, sports,
   animal). No harmful, political or war words.
3. For each candidate: coherence; fires_on agreement (share of fires_on tokens
   whose lowercase stem is in the up list or a close synonym list); lift on the
   8 lift prompts at doses +1, +2, +3.
4. Keep candidates with lift >= 3 at some dose. For each, run the demo prompt
   ("Tell me about your weekend.") at that dose through `/api/run` or
   `Engine.run`, and apply `coherent()` from `derive_llama_recipe.py`.
5. Rank by: passes both checks, then lift, then coherence. Write the shortlist
   and `cards.json`.
6. Record the best one with `record_sweep.py --rows L:N:1 --scales 0.5,1,2,3
   --prompt "Tell me about your weekend."` so replay mode can show it.

## Tests

- `test_search_ranks_by_both_checks` with a fake reader (no model): a
  candidate that fails steering never ranks above one that passes both.
- `test_shortlist_json_shape`.

## Acceptance checks

- `shortlist.json` exists with at least 50 scored candidates.
- At least one neuron with coherence >= 0.5, lift >= 3 and a coherent steered
  answer, **or** the report says none passed and names the trait fallback.
- `cards.json` serves in replay mode:
  `MODEL_LOBOTOMY_REPLAY=1 uv run python -m model_lobotomy.server`, then
  `curl localhost:7860/api/neuron/<L>/<N>` returns the card.
- The recorded sweep replays in the live UI.

## Done means

Report with the top 5, the numbers, the recommendation, total model time and
peak memory. Model lane booked and released in `model-lane.md`.

## Out of scope

UI. Changing reader math (file an issue to R1 instead).

## Risks

- Readable neurons sit in late layers, steerable ones in middle layers. Report
  the overlap honestly; the fallback is a trait direction with its evidence card.
- Lift numbers inflate on rare words (tiny base p). Require base_p >= 1e-4 for
  a word to count.
