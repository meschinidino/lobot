# FD Fluency-aware dose choice

Owner: backend agent. Lane: no-model for code and tests; one model
slot for the live runs. Depends on: PP (`pareto.FLUENCY_LABEL`).
Time: half a day.

## Problem

Create picks the dose on the tuning split. A dose is "coherent" when
the mean reply NLL rises by at most 1.0 nats over the base. The best
coherent dose by trait judge wins.

The PP report shows that this rule lets broken text through. Pirate's
frozen dose (1.5) writes "ol' ol' ol' ..." with fluency 0.017. Repeated
tokens are easy to predict, so their NLL is low. On the pirate tuning
table the rule rejected dose 1.0 (NLL +1.07) and kept dose 1.5 (NLL
+0.79).

## Rule (fixed before any live run)

`dose_rule = "nll+fluency-v1"`.

1. For every tuning cell (layer, dose), the judge also scores each
   reply with `pareto.FLUENCY_LABEL` ("written in fluent, coherent
   language"). The cell stores the mean as `fluency`.
2. A dose is coherent when both hold:
   - NLL rise over the base is at most 1.0 (unchanged).
   - Mean fluency is at least `FLUENCY_FLOOR = 0.5`.
3. The best coherent dose by trait judge wins, as before. The layer
   choice is unchanged.

Why 0.5: the judge score is P(yes). At 0.5 the judge says "fluent" as
often as "not fluent". The floor is the judge's own decision boundary,
not a value fit to pirate or cowboy.

The NLL check stays. It still catches gibberish with a high NLL. FD
changes one thing only.

## Output

- `report.json` `tuning.rule` records the rule name and the floor.
- Each `tuning.table` cell gains `fluency`.
- Patches made before FD keep their dose. FD does not rewrite them.

## Deliverables

- `trait_eval.coherent_doses`, `trait_eval.best_dose` (pure).
- `tune_layer_dose` scores fluency per cell.
- Unit tests with fakes.
- Report `docs/roadmap/reports/FD.md`.

## Acceptance

1. Unit suite passes. The rule has tests for: a fluent dose wins, a
   low-NLL broken dose is rejected, a high-NLL dose is rejected, and
   no coherent dose gives `None`.
2. Live on Llama 3.2 3B: re-run Create for pirate and cowboy (llama
   lane, full schedule, same seed), then `scripts/pareto.py`. The
   gate passes when, for both, the new frozen dose has final-split
   fluency at least 0.5 in `pareto.json`. The report gives the trait
   gain, the card result and the Pareto row for both, pass or fail.

## Known risk

A lower dose can give a smaller trait gain. Then the report card can
fail where it passed before. That is the honest result: the old pass
came from broken text that the trait judge still scored high.
