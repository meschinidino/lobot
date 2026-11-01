# GR Guard repair

Owner: backend agent. Lane: no-model for code and tests; one model
slot for the live runs. Depends on: GX screens, privacy v2,
`Guard.check_confirmed`. Time: two days.

## Goal

Today the guard only flags a patch. GR tries to repair it: keep the
behavior, remove the part of the direction that hurts safety, and
prove both with new evidence.

Basis: "Safety cost of steering vectors is separable and reducible"
(2026) finds one direction whose removal from a steering vector
restores safety at small cost to the effect. GR tests a cheaper form
of that idea: remove known safety directions, then fall back to a
lower dose.

## Safety directions

One unit direction per screen, at the patch layer, from mean
activation differences (last prompt token). The fit prompts are
disjoint from the guard screen prompts, so the guard stays a
held-out test.

| Screen | Positive side | Negative side |
|---|---|---|
| refusal | requests the model refuses | matched harmless requests |
| honesty | false premises, "answer truthfully" | the same, "agree with me" |
| privacy | planted secret, "never reveal it" | planted secret, "you may share it" |

## Candidates (order fixed before any measurement)

`project` removes the flagged screens' directions:
`unit' = normalize(v - Q Q^T v)`, with Q an orthonormal basis.

1. `project` at the patch dose.
2. `project` at 0.75x dose.
3. `raw` (original direction) at 0.75x dose.
4. `project` at 0.5x dose.
5. `raw` at 0.5x dose.

For each candidate: confirmed guard on the flagged screens only, and
the trait gain on the tuning split. The first candidate with clean
flagged screens and a positive tuning gain is chosen. Then the chosen
candidate gets the full confirmed guard (all enabled screens) and the
update_up report card on the final split, once.

## v2: gradient projection (added after v1 measured `unrepaired`)

v1 result: the mean-diff safety directions were almost orthogonal to
the patch directions (|cos| 0.004-0.055), so projection removed under
0.3% of the vector and changed nothing. v2 targets the safety cost
itself, to first order:

- For each fit prompt (the guard's situation, no safety cue), take the
  gradient of the safe reply's NLL ("I can't help with that.", "No,
  that's a common myth.", "I'm sorry, but I can't share that.") with
  respect to the patch layer's output, summed over positions, with the
  patch applied.
- Adding the patch vector changes that NLL by `v . g` to first order.
  `unit' = normalize(v - Q Q^T v)`, with Q spanning every gradient row
  of the flagged screens, makes that change zero on every fit prompt.
- Candidates, fixed before v2 ran: `gradient` at 1.0x, 0.75x, 0.5x the
  patch dose. The choice rule and final checks are the same as v1.
- `scripts/repair_patch.py --method gradient` (default) writes
  `repair_gradient.json`; `--method project` keeps v1 (`repair.json`).

## Output

- `repair.json`: safety-direction cosines, every candidate's result,
  the chosen one, its full guard and card.
- `knob.repaired-<method>.lobot` when a candidate is chosen. Its recipe has a
  `repair` block (method, removed screens, dose factor, cosine to the
  original). Its guard is the confirmed verdict, so the registry can
  publish it when it is clean.
- `status`: `repaired`, `unrepaired` (no candidate passed), or
  `clean` (nothing to repair).

## Deliverables

- `src/model_lobotomy/repair.py`: fit sets, `safety_directions`,
  `project_out` (pure), `repair_patch(...)`.
- `scripts/repair_patch.py JOBDIR`.
- `tests/test_repair.py` (fakes, no weights).
- Report `docs/roadmap/reports/GR.md`.

## Acceptance

1. Unit suite passes.
2. Live on Llama 3.2 3B: pirate (refusal flag) and noir (privacy
   flag). The gate passes when at least one yields `repaired` with a
   clean full guard and a passing update_up card. The report gives
   the numbers for both, pass or fail.

## Known risk

Random directions also leak secrets (privacy v2), so leaks can come
from the size of the push, not its direction. In that case `project`
fails and only the lower-dose candidates can pass. The report says
which candidate passed.
