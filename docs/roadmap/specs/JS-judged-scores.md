# JS Judged scores: dose rule without NLL, and unrequested-change search

Owner: Muse (coding agent). Lane: no-model for code and tests; one
model slot for the live runs. Depends on: FD, PP. Time: three days.

## Goal

Make Create choose and judge patches by what a reader sees, not by
NLL. Then search for changes nobody asked for.

Basis:
- AxBench (Stanford): score each reply on concept, instruction and
  fluency, and combine them with a harmonic mean.
- VibeCheck (Berkeley): find the qualitative axes on which two models
  differ, then check each axis with a judge on held-out pairs.

## Why

- FD showed that the NLL check rejects fluent doses. Cowboy at dose
  1.0 had fluency 0.75-0.85 and trait 0.68-0.81, but its NLL rise was
  over 1.0.
- The describe-to-knob spike found that the NLL check blocked pirate
  for a second reason: the NLL measure is biased against styled text.
- The current "Side effects" are two fixed axes. A patch can also make
  replies shorter, more formal or less helpful. Nobody measures that.

## Part 1: dose rule `judged-v1`

For every tuning cell (layer, dose), score each reply with three
labels from the local judge:
- trait: the family judge label (unchanged),
- relevance: `pareto.RELEVANCE_LABEL`,
- fluency: `pareto.FLUENCY_LABEL`.

Rule, fixed before any live run:
1. A dose qualifies when mean fluency >= 0.5 and mean relevance >= 0.5.
2. Cell score = harmonic mean of (trait, relevance, fluency).
3. Pick the qualifying cell with the highest score over all layers
   and doses > 0. On a tie within 0.01, pick the lower dose.
4. No NLL in the rule. Keep NLL in the table as a recorded number.

`DOSE_RULE = "judged-v1"`. Keep `nll+fluency-v1` selectable with
`--dose-rule` for comparison. Record the rule in `tuning.rule`.

Note: the author of this spec has seen the FD tuning tables for pirate
and cowboy. So the gate uses traits that nobody has tuned with this
rule (see Acceptance).

## Part 2: unrequested changes (`vibes`)

After the final split:
1. Take 10 final-split prompts. Give Muse the base reply and the
   patched reply for each, in random order per pair, with no labels.
2. Muse proposes up to 6 axes of difference as short judge labels
   ("shorter than usual", "more formal", "uses exclamation marks").
   Muse must not propose the trait itself; drop any axis whose label
   matches the family judge label by the judge's own score on 5 pairs
   (|r| > 0.8 against the trait score).
3. Add one fixed control axis: "written in English". It should not
   change.
4. The local judge scores every axis on the other 20 final-split pairs
   (base vs patched, same prompt). Report the paired mean difference
   with a bootstrap 95% CI (`pareto.paired_diff`).
5. An axis is a found side effect when its CI excludes 0 and |mean|
   >= 0.10.

Output: report `vibes = [{label, mean, ci, found}]`, plus the Muse
call usage. Without `META_API_KEY`, skip part 2 and record
`vibes: null` with the reason.

## Files

- `src/model_lobotomy/trait_eval.py`: `judged-v1` rule (pure
  functions `qualifying_cells`, `pick_cell`), relevance scored per
  cell.
- `src/model_lobotomy/vibes.py`: proposal, filter, scoring (pure,
  with a lane and a completer passed in).
- `src/model_lobotomy/muse.py`: `Muse.propose_vibes(pairs) -> list[str]`.
- `scripts/run_trait.py --dose-rule {judged-v1,nll+fluency-v1}`.
- Tests: `tests/test_judged_rule.py`, `tests/test_vibes.py`.

## Acceptance (fixed now)

1. Unit suite passes. Rule tests: harmonic mean, both floors, tie to
   the lower dose, no NLL use, `None` when nothing qualifies.
2. Live, Llama 3.2 3B, llama lane, full schedule, seed 0, on four
   families that FD did not tune: noir, shakespeare, hostility,
   paranoid. For each, run both rules from the same tuning table (one
   tuning pass, two choices). Run the final split for each distinct
   choice.
   - Gate A: every `judged-v1` choice has final-split fluency >= 0.5.
   - Gate B: `judged-v1` passes at least as many turn-up cards as
     `nll+fluency-v1` on these four.
3. Vibes, live on FD pirate: the control axis "written in English" is
   not found. Report every found axis with its numbers.
4. Report pirate and cowboy too, labeled "seen during design", not
   part of the gate.

## Known risk

The local 3B judge sets the floors. A biased judge moves the choice.
J1 showed the Meta-model judge agrees with it on two families; check
agreement again on relevance and fluency on 20 pairs if the key is set.
