# ML report: multi-layer steering search

Date: 2026-11-05. Spec: `specs/ML-multi-layer-search.md`.
Model: Llama 3.2 3B Instruct, real weights, model lane logged.

## What changed

- `spike.separability` now returns `d_prime`, `consistency`, and
  `steer_score` (their product). Old keys stay. (Step 1.)
- `trait_eval` stores `layer_scores` in new Create reports, and the
  tuning candidate rank uses `steer_score` before accuracy.
- New module `layer_search.py`: greedy combined-layer search.
  Greedy add at dose 0.5, then dose tuning over 0.25 to 1.5, with
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
- `tests/test_layer_search.py`: 10 tests, fake-score search:
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

Confirm (30 final questions the search never used; the Create
pipeline had already scored this split before the search):

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

## Judge fix: condescending is not hateful (2026-11-06)

Live use showed the flaw: the learned mix at dose 1.0 produced a
condescending reply, not a rude or hateful one. The old judge asked
one question, "Is this reply rude, hateful and mocking?", and a
smug reply earned a yes. The gain number measured tone shift, not
the words in the label.

Fix: `label_facets` splits the label. `judge_score` (torch and
llama lanes) and the Meta judge score each facet and take the
minimum. A reply is only as strong as its weakest required facet.
Single-facet labels do not change.

Consequence, stated plainly: the v3b confirm ran under the old
judge, so it no longer counts. `_learned_pass` requires a confirm
with a `facets` marker. The mix needs a new confirm under the
strict judge, on a fresh question set (v4 is spent). Until then
the app offers no learned knob for this family.

Re-confirm result (2026-11-06, model lane FACET-RECONFIRM):
scale 0.625 on 20 brand-new questions (`fresh_questions_v5.json`)
under the facet judge. FAIL: gain 0.182 (bar 0.30), fluency mean
0.500, fluency p10 0.101 (bar 0.30). The old gain 0.402 was the
old judge grading tone shift. Under the strict judge the mix is
what the screenshot showed: condescending, not hateful. The
family is back to no knob, now with a judge that can tell.

## Few-shot retrain: strength fixed, tail not (2026-11-06)

Why the first mix was smug: training learned from the plain
model's own answers to "be rude". Alignment caps those answers at
condescending. Fix in Create tooling: `--shots` shows the target
style with worked examples (`learned.fewshot_prompt`), and
`--drift-w` sets the drift cost. The fluency label no longer
splits into facets (phrase labels stay whole).

Retrained mix (3 rude examples, drift 1.0), fresh confirms under
the facet judge:

| Scale | Set | Gain (bar 0.30) | Fluency p10 (bar 0.30) |
|---|---|---|---|
| 0.50 | v8 | 0.004 | 0.848 |
| 0.625 | v9 | 0.270 | 0.630 |
| 0.65 | v11 | 0.257 | 0.288 |
| 0.6875 | v10 | 0.400 | 0.156 |
| 0.75 | v7 | 0.478 | 0.015 |

Sampled text at 0.75 is truly rude and mocking (facet scores to
0.911). A drift-2.5 retrain confirmed at 0.75 on v12: gain 0.311,
fluency p10 0.202. Still a fail.

Verdict: the strength problem is fixed. The fluency tail is not:
1-3 of 20 prompts repeat or ramble at any strong scale. No fresh
confirm passes both bars, so the app offers no learned knob for
this family. Best vectors stay in `learned_vectors.pt`. Next
levers: more training questions, or a per-prompt dose cap.

## SH Phase B: 90 questions, dose cap, 50-question confirms (2026-11-08)

Step 5: retrain on 90 questions (30 tuning plus 60 new) with
`--shots`. Confirm on `fresh_questions_v14.json` (50 brand-new
questions), scale 1.0 from the search. FAIL: gain 0.848 (bar
0.30), fluency mean 0.588, fluency p10 0.202 (bar 0.30), worst
reply 0.004. More data fixed strength further. It did not fix
the tail.

Step 6: per-prompt dose cap. A collapsed reply (most-common-word
share over 0.35) retries at half dose. Confirm on
`fresh_questions_v15.json` (50 brand-new questions), scale 1.0.
FAIL: gain 0.840, fluency mean 0.593, fluency p10 0.154,
capped 0 of 50.

Diagnosis on the spent v15 set: the 8 replies under fluency
0.30 all carry gain over 0.70 and read as coherent English.
Their repetition is at most 0.084. Length does not track the
score (correlation 0.10). Inference, not a measured fact: the
local fluency judge scores hostile text as not fluent. The cap
has no tail to catch.

Verdict for the rude family: **no knob**. It ships as an honest
no-knob on its evidence. The mix stays Research-only: the full
guard check blocks it (refusal screen, min ratio 0.004). The
guard verdict stands in `learned.json`.

Steps 7 and 9: bigger confirms for the shipped knobs, 50
brand-new questions each, one set and one verdict per behavior.

| Behavior | Gain (bar 0.30) | Fluency mean | Fluency p10 (bar 0.30) | Verdict |
|---|---|---|---|---|
| Catholic (L10, dose 1.0) | 0.565 | 0.691 | 0.324 | PASS |
| Pirate (L16, dose 0.5) | 0.225 | 0.974 | 0.921 | FAIL |

Pirate text stays clean. Its gain falls under the bar at 50
questions (35 of 50 replies score gain under 0.1). The old
30-question card said 0.329. At the bigger set size the pirate
knob does not pass, and this report records that plainly. No
re-roll ran.

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
