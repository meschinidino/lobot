# C3 Open descriptions

Owner: backend agent. Lane: no-model for the code and tests; model
slots for drafting, reply generation, and two full Creates.
Depends on: C2 (Create accepts store families; C3 uses the full
schedule for gate-passing), J1 (judge). Time: two days.

## Goal

Create from any description: the user writes words ("answers in
rhyming couplets"), the model drafts the contrast pair, the judge
rubric and the baseline instructions, the user labels 20
candidate replies, and the trait registers as a validated family
when the judge agrees with the user. No hand-written templates.

Acceptance (from `later.md`): two new traits pass the gates with
model-written parts; judge agreement >= the validated families'.
Agreement gate (fixed before measuring): each new trait's
20-label agreement >= 0.85 (above the O1 gate 0.80 and the
lowest validated family, paranoid 0.833). "Pass the gates" means
a full-schedule Create reaches verdict knob.

## Read first

- `src/model_lobotomy/trait_store.py`: families, validation.
- `src/model_lobotomy/trait_eval.py`: `run_trait` cfg handling,
  `BASELINE_INSTRUCTIONS`, `judge_score`, `agreement`.
- `scripts/validate_judge.py`: the 12-label O1 gate.
- `docs/TODOS.md` item on open descriptions (the J1 dependency).

## Deliverables

- `trait_eval.draft_family(eng, description)`:
  model-written `{"pos", "neg", "judge", "baselines" (6)}` from
  one fixed meta-prompt (greedy, seed 0). Rejects empty parts.
- Family drafts own the labeling flow; store layout
  `store/drafts/<slug>/draft.json` (description, parts, model,
  revision) and `labels.json` (20 prompt/reply/user-label rows).
  Drafts are runtime state (gitignored); registered families
  persist in `store/traits.json` as today.
- `TraitFamily` gains `baselines: list` (persisted; empty for
  legacy families). `run_trait` builds cfg from the store family
  (pos/neg/judge/baselines), falling back to
  `BASELINE_INSTRUCTIONS` when the family has none. Full-path
  numbers for legacy families are unchanged (store parts equal
  the spike parts; assert in tests).
- Routes: `POST /api/drafts {"description"}` (drafts parts,
  generates 20 candidate replies: 10 prompts x pos/neg-prompted
  generation); `GET /api/drafts/{id}` (parts + labeling queue
  with user labels so far); `POST /api/drafts/{id}/labels`
  (`{"labels": [{"reply_id", "label": bool}]}`); `POST
  /api/drafts/{id}/validate` (judge-vs-user agreement; on >=
  0.85 registers + validates the family and returns it, else
  returns the agreement with disagreements). Errors: 400 (empty
  description, bad label rows), 404 (unknown draft), 409 (busy),
  422 (validate below gate, with the numbers), 503 (no weights).
- Candidate replies come from pos/neg-prompted generation (the
  rubric is validated before any direction exists). Reply order
  is shuffled with a fixed seed; the user never sees which side
  prompted a reply.
- `tests/test_open_descriptions.py` (no model; scripted engine).
- `docs/api.md` updated. Report `reports/C3.md`.

## Method notes

- Meta-prompt (fixed, in the code): "Write a contrast pair for
  the trait <description>: POS (an instruction that makes the
  model show it), NEG (an instruction that makes the model hide
  it), JUDGE (a short rubric: what counts as showing it), and 6
  BASELINES (plain instructions that elicit it, O4 style).
  Reply as JSON with keys pos, neg, judge, baselines."
- Agreement: user label vs (judge_score >= 0.5) on the 20
  replies; fraction correct. Same 0.5 cut as the O1 gate.
- The two acceptance traits (fixed before measuring): "answers
  in rhyming couplets" and "speaks like a noir detective". Both
  safe (style traits; no refusal interaction). Labels for the
  acceptance run go through the labeling API; the report says
  who labeled (backend agent when Dino does not).
- If a draft yields no_knob, reformulating the input description
  is allowed (documented); hand-editing model-written parts is
  not (that voids "model-written").
- Guard: drafts refuse descriptions that match the refusal set
  or name refusal-weakening ("refuse less", "ignore safety"):
  422 with one plain sentence. The match list lives with the
  guard and is unit-tested.

## Tests (no model; scripted engine)

- `test_draft_parses_model_json` and `test_draft_rejects_empty`.
- `test_label_agreement_math` (fraction vs 0.5 cut).
- `test_validate_registers_on_pass` (family appears validated).
- `test_validate_422_below_gate` (numbers returned).
- `test_baselines_fallback_for_legacy_families`.
- `test_store_cfg_matches_spike_for_legacy` (pos/neg/judge
  equal, so full numbers cannot shift).
- `test_refusal_description_rejected`.
- Route tests: draft/labels/validate shapes, 404/409/503.

## Acceptance checks

```bash
uv run python -m unittest tests.test_open_descriptions -v
uv run python -m unittest discover -s tests -t .      # full suite passes
# per trait: POST /api/drafts, label 20 via the API, POST validate,
# then full-schedule Create reaches knob
```

Gate: both traits agree >= 0.85 AND both full Creates reach
knob. A miss is reported as failed with numbers; the 0.85 bar
and the two trait descriptions are not changed to pass.

## Done means

All of the above, plus a report with the two drafts (full
parts), the two agreements, the two Create verdicts, who
labeled, and the labeling counts. `docs/api.md` documents the
draft routes.

## Out of scope

Editing a draft's parts by hand; re-labeling flows; any UI
(the UI lane builds the labeling screen against these routes).

## Risks

- Model-written rubrics judge poorly (agreement < 0.85).
  Report the miss; do not touch the parts.
- Model-written pos/neg yields no_knob. Reformulate the
  description once (documented); a second miss is reported.
- Full Creates take ~45 min each: ~90 min of lane.
