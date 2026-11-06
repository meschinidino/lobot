# ML report: multi-layer steering search

Date: 2026-11-05. Spec: `specs/ML-multi-layer-search.md`.
Model: Llama 3.2 3B Instruct, real weights, model lane logged.

## What changed

- `spike.separability` now returns `d_prime`, `consistency`, and
  `steer_score` (their product). Old keys stay. (Step 1.)
- `trait_eval` stores `layer_scores` in new Create reports, and the
  tuning candidate rank uses `steer_score` before accuracy.
- New module `layer_search.py`: greedy combined-layer search.
  Greedy add at dose 0.5, then dose tuning over 0.25 to 1.0, with
  the fluency-first-v3 gates at every step. A fresh-prompt confirm
  applies the pass gates. (Step 2.)
- New CLI `scripts/combine_search.py <family> <job-dir>`. It ranks
  layers, searches on 10 tuning questions, confirms on 30 final
  questions, and writes `combine.json` next to the family report.

Note: the first version of step 2 used the file name `combine.py`.
That name already held the CB combined-patch checks. The same
mistake hit `tests/test_combine.py`. Both files were restored from
git in the same session. The search moved to `layer_search.py` and
its tests to `test_layer_search.py`. No CB behavior changed. The
suite count proves the restore: 829 tests now, the old 818 plus
the 11 new ML tests.

## Tests (fakes, no weights)

- `tests/test_layer_score.py`: 4 tests. Score math on toy tensors,
  plus `layer_scores` present with the right keys in a fake Create
  report.
- `tests/test_layer_search.py`: 7 tests, fake-score search:
  clean adds kept, breaking adds rejected, dose tuning, call cap,
  fresh-drop confirm fail (the layer 12 case).
- Full suite after v2: 831 tests, OK, 6 skipped. (829 after v1;
  the 2 probe tests are the delta.)
- `uvx pyflakes src scripts`: clean.

## Live run (real weights)

Family `be-rude-hateful-and-mocking-2`, Llama lane, 28 score calls.

Layer ranking by `steer_score`, top 5: L16 36.664, L17 34.175,
L20 33.899, L27 33.875, L19 32.639.

Acceptance check from the spec: **FAILED, by one place.** Layer 12
ranks 15 of 28. The spec asked for the top half (14). The gate
stays as written; this line reports the miss.

Why it missed: consistency is 1.000 on every layer. The positive
and negative instructions split the captures so hard that the
per-prompt sign test adds no signal. The rank is the d-prime rank.
And d-prime measures how well a layer *reads* the trait. It does
not measure how cleanly a layer *steers*. The best clean steering
layers in the sweep (12, 10, 13) rank 15, 17, and 12.

Search steps (10 tuning questions):

1. Start: gain 0.000.
2. Add L16 at 0.5: gain 0.090, worst fluency 0.871.
3. Add L27 at 0.5: gain 0.343, worst fluency 0.981.
4. No third add helped. Dose tuning found no better dose.

Confirm (30 final questions the search never used):

- Gain 0.260. Bar: 0.30. **Fail, close.**
- Fluency mean 0.926, worst decile 0.769. Fluency gates pass.
- One answer scored fluency 0.004. The gates read the decile, not
  the single worst answer, so this shows in the data but does not
  fail the gate.

Verdict: no combined knob for this family yet. But the mix beats
every single layer on fresh questions (best clean single: 0.199 on
search questions, layer 12's 0.338 fell to 0.129 fresh). Two weak
clean pushes added up, as the spec hoped. They reached 0.26 of the
0.30 bar without breaking the text gates.

## v2 run (same day, spec amendment written before the run)

Probe ranking (28 layers alone at dose 0.5, 5 tuning questions):
L11 0.570, L9 0.545, L12 0.431, L10 0.425, L13 0.369. The v2 gate
asked for layer 12 in the top 14. It ranks 3. **Pass.** The probe
top 8 matches the sweep's clean steering layers. The read score
never did that.

Greedy then picked a single layer: L11 at 0.5. Search gain 0.465,
worst fluency 0.872 on the 10 tuning questions. No add helped.
Dose tuning found no better dose. 21 search calls.

Confirm on the 20 brand-new questions (`fresh_questions_v2.json`,
written before the run, never searched):

- Gain 0.572. Bar 0.30. **Pass, for the first time on fresh
  questions.**
- Fluency mean 0.733, worst decile 0.026, worst answer 0.017.
  The worst-decile gate (0.3) **fails**.

Verdict v2: FAIL on fluency. The effect is real on new questions
now. The text still breaks on the hardest tenth of answers.

So the wall moved. Single layers and the v1 mix could not reach
the gain bar on fresh questions. This one reaches the gain bar
and breaks text. The remaining problem is the fluency tail, not
the effect.

## v3 run: learned vectors (2026-11-06)

Method: the v3 amendment in the spec. 8 layers, 30 tuning
questions, 90 training steps. Positive NLL fell from 2.98 to
0.95. Drift KL fell from 4.40 to 0.28. Full curve:
`learned_history.json` in the job dir.

Search scores (10 tuning questions): scale 0.5 gained 0.305 with
clean text; scale 0.75 gained 0.881, worst fluency 0.719.

Confirm at scale 0.75 on the v3 fresh set: gain 0.819, mean
fluency 0.637, worst-decile fluency 0.113. **FAIL** on the tail.

Exploration on that spent set: scale 0.5 gained 0.118 clean;
scale 0.625 gained 0.530, worst decile 0.531.

v3b confirm, scale fixed at 0.625 before the run, 20 brand-new
questions (`fresh_questions_v4.json`):

- Gain 0.402 (bar 0.30). **Pass.**
- Mean fluency 0.837 (gate 0.5). **Pass.**
- Worst-decile fluency 0.392 (gate 0.3). **Pass.**
- Worst single answer 0.162. The gate reads the decile, so this
  shows in the data without failing the run.

Verdict: **PASS.** Learned vectors are the first method to pass a
fresh confirm for this family. Searching (v1, v2) never did. The
drift cost is the likely reason, but that is a hypothesis: no
drift-off ablation has run. What the data does show: training
pushed hard, stayed clean in the mean, and at scale 0.625 the
tail held on new questions.

Two bugs found on the way, both fixed before the scored runs:
batches could hold only one side of a pair (fixed: batch by
pair), and the scorer imported the CB `combine` module instead of
`layer_search` (fixed; the trained vectors were resumed from
disk, not retrained).

## Learned mixes in the app (2026-11-06)

The passed v3b mix now runs through the app. `patch.py` holds it as a
`kind: "multi"` patch (`save_multi_patch` / `load_patch`): one raw
vector per layer, the same hash pins as single patches. `learn_vectors.py`
writes `learned.lobot` on a passed confirm. The server reads the pass
from `learned.json` (`_learned_pass`), offers the trait as a learned
knob (dose 1.0 = the tested mix at scale 0.625), and streams it
through `/api/update` and chat steering like any knob.

Live check on the real app: dose 1.0 shifted the tone meaner and kept
the text clean in Research mode. With the guard on, the full guard
BLOCKED the mix (min_ratio 0.0): the mean push also weakens refusals.
The verdict sits in `learned.json` and in the patch meta. The shelf
shows `GUARD` in normal mode and `LEARNED · GUARD` in Research mode.
A trait pass is not a safety pass. Both gates stay separate.

## Known gaps

- `steer_score` ranks reads, not steering. A v2 score needs a
  steering signal. See `docs/TODOS.md`.
- Greedy stopped at two layers because the third add must gain
  0.02 to count. A smaller step, or a finer dose grid near the
  winner, may close the last 0.04. Untested.
- Combined knobs still have no patch file format. The confirm
  failed, so per the spec that work stays deferred.
- `combine.json` has no UI. Evidence pages do not show mixes yet.

## Files

- Code: `src/model_lobotomy/spike.py`,
  `src/model_lobotomy/trait_eval.py`,
  `src/model_lobotomy/layer_search.py`,
  `scripts/combine_search.py`.
- Data: `combine.json` and the ranked layers sit next to the
  family report in the job dir.
- Docs: the ML spec, this report, `docs/reference/scripts.md`,
  `CHANGELOG.md`, `docs/TODOS.md`.
