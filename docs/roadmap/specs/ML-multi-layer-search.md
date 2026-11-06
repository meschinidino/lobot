# ML: multi-layer steering search

Owner: Muse (coding agent). Lane: no-model for code and tests; model
lane for the live run. Depends on: the direction port
(`directions/`), the dose rules (`dose_rules.py`), and the all-layer
sweep evidence on `be-rude-hateful-and-mocking-2` (2026-11-05).
Time: one day.

## Why this work exists

Create steers one layer at a time. A full sweep tested that claim on
the evil-and-mean family (`be-rude-hateful-and-mocking-2`, Llama 3.2
3B):

- 28 layers, doses 0.5, 1.0, 1.5, 2.0: 112 cells, 10 held-out
  questions per cell.
- 70 cells reached the gain bar (0.30, `evaluation.MIN_GAIN`). 69 of
  them broke at least one answer (worst fluency 0.114 or less).
- One cell passed both on the search questions: layer 12 at dose 0.5,
  gain 0.338, worst fluency 0.679.
- On 20 fresh questions, that cell gained 0.129 (CI95 0.04 to 0.239).
  Its text stayed clean. The good search score was luck.

Two lessons follow:

1. No single-layer patch in the tested range is both strong and
   clean for this family.
2. A search that picks the best of many cells must confirm on
   questions the search never saw. Search scores lie. Fresh scores
   decide.

A combined patch may still work. Small pushes on several layers may
add up while each push stays weak. Brute force cannot test the mix
space: 28 layers at 5 dose choices each (off plus 4 doses) is 5^28
mixes. This workstream adds a search that stays small.

## Published basis

- LayerNavigator (NeurIPS 2025) names the same problem:
  multi-layer steering faces a combinatorial explosion. It scores
  each layer by discriminability and consistency. It reuses the
  activations from direction building, so the score costs no model
  runs. <https://papers.nips.cc/paper_files/paper/2025/hash/9280bfb61f159c768dfdeb729a41ccd0-Abstract-Conference.html>
- Sweep Less, Steer More shows a cheap geometry score (d-prime)
  finds the good layers and removes the layer sweep.
  <https://arxiv.org/abs/2611.01496v1>
- GRACE treats steering as budgeted search and cuts the test count
  by about 40% with Bayesian optimization. That is step 3 material,
  not this spec. <https://overfitted.cloud/pdf/2605.16362v2>

## Step 1: a layer score that can rank

Today `extraction_seps` stores only held-out accuracy (`acc`). On
the evil family, 25 of 28 layers tie at 1.000. The score cannot rank.

Add to `spike.separability` (and so to the direction port's `seps`):

- `d_prime`: the held-out separation along the direction, in pooled
  standard deviations. This is the `effect` value the function
  already computes. Give it its published name.
- `consistency`: the share of paired prompts where the positive
  capture scores above the negative capture along the direction.
  Captures are paired by prompt, so this is a per-prompt sign test.
- `steer_score`: `d_prime * consistency`. One number to rank by.

`separability` keeps its current keys (`acc`, `effect`, `unit`).
New keys only add data; no caller changes shape.

`trait_eval` stores the new fields in the report as `layer_scores`,
one entry per layer: `acc`, `d_prime`, `consistency`,
`steer_score`. The Create flow computes them from the held-out part
of the extraction captures, the same split `seps` already uses.

Acceptance for step 1 (fixed before the live run):

- Unit tests pin the math on toy tensors: a clean split scores
  consistency 1.0; a half-flipped split scores 0.5; `d_prime` matches
  a hand-computed value.
- On the evil family report, `layer_scores` ranks layer 12 in the
  top half. (Its single-layer cell was the only clean strong one;
  a score that ranks it last is wrong.)

## Step 2: greedy combined search

New module `src/model_lobotomy/layer_search.py`. It searches mixes of
per-layer vector pushes. The engine already applies many vector
edits in one run (`EditSet.vector` is a tuple), so no engine change
is needed.

### Interface

- `score` is an injected callable:
  `score(rows, prompts) -> {"gain", "fluency_mean", "fluency_min",
  "fluency_p10"}`. `rows` is a tuple of `(layer, dose)` pairs.
  Tests inject a fake score. The live adapter builds the real one
  from the engine, the local judge, the family direction, and the
  layer medians.
- `greedy_search(score, ranked_layers, prompts, ...) -> result`.
  Pure logic. No engine import.
- `confirm(score, rows, fresh_prompts) -> verdict`. Runs the winner
  once on fresh prompts and applies the pass gates below.

### Search rules (fixed before the live run)

- Candidates: the top `TOP_K = 8` layers by `steer_score`.
- Add dose: 0.5 per layer, the lowest sweep dose. Weak pushes keep
  text clean.
- Greedy add: start from no layers. Try each candidate layer once.
  Keep the best add only if gain improves by at least
  `MIN_IMPROVE = 0.02` and the fluency gates hold. Repeat until no
  add helps or the mix has `MAX_LAYERS = 5` layers.
- Dose tune: two passes over the mix. For each layer, try doses
  0.25, 0.5, 0.75, 1.0. Keep the best dose by gain, with the
  fluency gates held. This is hill climbing on the mix.
- Fluency gates at every step: mean fluency at least
  `FLUENCY_FLOOR` (0.5) and worst-decile fluency at least
  `FLUENCY_P10_FLOOR` (0.3), the `fluency-first-v3` rule. A step
  that breaks a gate does not count, even with a high gain.
- Cost guard: at most `MAX_SCORE_CALLS = 200` score calls. The
  search stops and reports the best mix so far when it hits the cap.

### Pass gates (fixed before the live run)

The confirm step passes only if all hold on fresh prompts:

- mean gain at least `MIN_GAIN` (0.30),
- mean fluency at least `FLUENCY_FLOOR`,
- worst-decile fluency at least `FLUENCY_P10_FLOOR`.

The search split never confirms. Tuning prompts search. Final
prompts confirm. The report states both scores side by side.

### Live adapter and CLI

`scripts/combine_search.py <family> <job-dir>`:

1. Reads the family report (`splits`, `layer_scores` when present;
   else ranks by `extraction_seps` accuracy and says so).
2. Builds the direction with the direction port on the extraction
   split, as the sweep scripts do.
3. Searches on the first 10 tuning questions (cost: each score call
   runs every search question). Confirms on all 30 final questions.
   Plain answers are cached once per question set, so a score call
   generates only the edited answers.
4. Writes `combine.json` next to the report: ranked layers, every
   search step (mix, gain, fluency), the best mix, and the confirm
   verdict with both scores.
5. Takes the model lane like every live run: log start and end in
   `docs/roadmap/model-lane.md`, stop the app server first, run one
   model job at a time.

First live target: `be-rude-hateful-and-mocking-2` on Llama 3.2 3B.
The verdict may be fail. A fail with numbers is a valid result.

## v2 amendment (2026-11-05, written before the v2 live run)

The v1 live run showed two things. The layer score ranks reads,
not steering (layer 12 ranked 15 of 28). The mix reached 0.260 on
fresh questions, 0.04 under the bar. v2 changes the search, not
the pass gates. The pass gates in this spec stay as written.

1. **Steering probe score.** Before greedy, score each of the 28
   layers alone at dose 0.5 on 5 tuning questions. Rank by clean
   gain: the gain when the fluency gates hold, else the gain minus
   1.0. Broken strength ranks under clean weakness. Gate for this
   step, fixed now: layer 12 must rank in the top 14 by probe.
2. **Push harder.** The add threshold drops from 0.02 to 0.01.
   The dose grid grows to 0.25, 0.5, 0.75, 1.0, 1.25, 1.5.
   Candidates stay the top 8, now by probe rank.
3. **Brand-new confirm questions.** The final split is spent (the
   v1 confirm used it, and the sweep used its first 10). Twenty
   hand-written questions, stored in the job dir as
   `fresh_questions_v2.json`, were written before this run. No
   search step sees them. Only the confirm does.

The v2 CLI writes `combine_v2.json`. It does not overwrite the v1
`combine.json` evidence.

v2 results (2026-11-05, `reports/ML.md`): probe gate **passed**
(layer 12 ranks 3 of 28). Winner: L11 at 0.5 alone. Confirm on the
brand-new questions: gain 0.572 (bar 0.30, pass), worst-decile
fluency 0.026 (gate 0.3, **fail**). Verdict: FAIL on fluency.

## Out of scope

- Learned steering vectors trained by gradient (step 3). It needs a
  differentiable objective and a drift cost. See `docs/TODOS.md`.
- Bayesian optimization of the mix (GRACE). Greedy plus hill
  climbing ships first; the step log shows if it wastes calls.
- A patch file format for combined knobs. `knob.lobot` holds one
  layer today. A passing mix raises that design question; a failing
  one does not. See `docs/TODOS.md`.
- UI. The result lands in `combine.json` and the report file for
  this workstream first.
- Neuron-level mixes. This spec combines layer directions only.

## Tests

- `tests/test_layer_score.py`: the score math on toy tensors.
- `tests/test_combine.py`: fake-score search tests.
  - Greedy keeps an add that helps and stays clean.
  - Greedy rejects an add that breaks the fluency gate.
  - Dose tune moves a dose and keeps the better one.
  - The call cap stops the search.
  - Confirm fails a winner whose fresh gain is under the bar (the
    layer 12 case, as a fixture).
- Existing suite must stay green. `separability` gains keys, so
  direction and trait tests run too.

## Acceptance

- [x] Step 1 unit tests pass; `layer_scores` appears in a replay or
  fake Create report with the expected keys.
- [ ] Step 1 live check: layer 12 ranks in the top half on the evil
  family. **FAILED**: it ranks 15 of 28 (see `reports/ML.md`). The
  gate stays as written here; the miss is reported, not hidden.
- [x] Step 2 unit tests pass with a fake score. No weights loaded.
- [x] Full suite passes; `uvx pyflakes src scripts` is clean.
- [x] Live run on the evil family finishes and writes
  `combine.json`. The report (`reports/ML.md`) states the ranked
  layers, the best mix, the search score, and the confirm score.
  Confirm verdict: FAIL (gain 0.260, bar 0.30).
- [x] `CHANGELOG.md` and `docs/TODOS.md` updated in the same change.

## Report

`docs/roadmap/reports/ML.md`: what changed, test counts, the live
search log in short form, the confirm numbers, and known gaps.
Say plainly which parts ran on real weights and which ran on fakes.
