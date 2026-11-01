# SA Side-effect atlas and forecast

Owner: Muse (coding agent). Lane: no-model for code and tests; one
model slot for the atlas run. Depends on: PP, a store with at least 6
knobs. Time: two days.

## Goal

1. Measure what each patch does to every other behavior and to every
   guard screen: the atlas.
2. Predict those side effects from the unsteered model, before any
   generation: the forecast.

Basis: "Forecasting side effects of activation steering" (2026). It
predicts off-target effects from the geometry of the unsteered model.

## Part 1: atlas

1. Patches: every knob in the store with a passing turn-up card, on
   Llama 3.2 3B. List them in the report before the run. Need >= 6.
2. Prompts: 20 neutral prompts, disjoint from every family's splits
   (add `ATLAS_PROMPTS` to the store module).
3. For each patch i at its frozen dose, generate replies (llama lane).
   Score each reply with every family judge label j and with
   relevance and fluency. Run the guard screens (dev split) once per
   patch.
4. `M[i][j]` = mean judge score with patch i minus the base mean for
   label j. Store with bootstrap CIs.
5. Output: `artifacts/atlas/<model>/atlas.json` (git-ignored) and a
   recorded copy for replay.

## Part 2: forecast

Predictors, all from the unsteered model, fixed now:
1. `cos`: cosine of direction i and direction j.
2. `proj`: (unit_i . mean activation difference of family j at the
   layer of i) x dose_i x median scale. The activation difference
   comes from family j's contrast prompts (cached capture).
3. Guard forecast: `proj` of unit_i on each GR gradient row set
   (refusal, honesty, privacy), max abs value.

Evaluation:
- Spearman correlation between each predictor and measured `M[i][j]`
  over all off-diagonal pairs, with a bootstrap CI over patches.
- For guard screens: the forecast ranks patches; report recall of the
  flagged patches in the top half.

## Use

- The evidence page (U4 follow-up) shows "likely side effects": the
  top 3 other behaviors by forecast, with the measured value when the
  atlas has it.
- The forecast must **not** skip any guard screen in this version.
  Publishing still needs the full confirmed guard.

## Files

- `src/model_lobotomy/atlas.py`: matrix build (lane passed in),
  predictors (pure), correlation.
- `scripts/atlas.py --lane llama`.
- Tests: predictors on fake directions, matrix shape, CI, Spearman.

## Acceptance (fixed now)

1. Unit suite passes.
2. Atlas built for >= 6 patches, with every cell's CI.
3. Forecast gate: Spearman of `proj` with measured off-diagonal
   effects >= 0.5, with the CI lower bound > 0. Report `cos` as the
   simple baseline. If `proj` does not beat `cos`, say so.
4. Guard forecast: report recall; no gate.

## Known risk

With only 6-10 patches, the correlation CI is wide. Report it as is.
