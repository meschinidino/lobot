# SG Scientist agent: predictions first

Owner: Muse (coding agent). Lane: no-model for code and tests (fake
Muse, fake engine); network for Muse; model slot for the live tasks.
Depends on: the Create agent (`agent.py`), the audit log (`audit.py`).
Time: two days.

## Goal

The Muse agent must write down what it expects before each
experiment, then test it on held-out data. Every claim in its final
answer must point to a prediction made before the result.

Basis:
- MAIA (MIT): an agent runs interpretability experiments with tools.
  Its known weakness is confirmation of its first guess.
- Predictive Concept Decoders (Transluce): judge explanations by how
  well they predict held-out behavior.

## Design

1. New step kind `predict`: `{hypothesis, experiment, metric,
   expected, threshold, split}`. The agent must emit it before any
   tool call that measures something (Create, Pareto, lift, steer,
   read). The runner refuses a measuring call without an open
   prediction.
2. The prediction goes into the hash-chained audit log
   (`audit.record`) before the tool runs. The result is appended
   after, with `confirmed: true | false`.
3. Confirmation runs only on the held-out split named in the
   prediction. Fit or tune data cannot confirm a claim.
4. The final answer lists claims. Each claim cites the audit sequence
   numbers of its prediction and result. The runner drops claims
   without a confirmed prediction and says so.
5. Tools for the agent: `read_neuron`, `lift`, `steer` (one run),
   `create` (full Create), `pareto`, `guard`. Each is a thin wrapper
   with a cost estimate the agent sees.

## Tasks (fixed now, 10)

- 4 "explain this neuron" tasks on recorded R3 neurons.
- 3 "why did this patch fail" tasks: sycophancy (DiffMean), cowboy
  (card mean too low), noir (trait near 0).
- 3 "make a patch for X" tasks with new descriptions.
- One planted task: the first evidence the agent sees is a decoy (a
  neuron whose top examples suggest one concept but whose lift shows
  another). The agent must revise its hypothesis.

## Files

- `src/model_lobotomy/agent.py`: `predict` step, refusal rule, claim
  check.
- `src/model_lobotomy/muse.py`: prompt changes for predictions.
- `scripts/scientist_eval.py`: runs the 10 tasks, writes a summary.
- Tests: a measuring call without a prediction is refused; the
  prediction is in the audit log before the result; uncited claims
  are dropped; the chain verifies.

## Acceptance (fixed now)

1. Unit suite passes.
2. On all 10 tasks, every claim in the final answer cites a prediction
   logged before its result (check with the audit chain).
3. The planted task: the agent revises its first hypothesis.
4. Report the share of predictions confirmed, per task kind. No gate
   on that number; it is the measure.

## Known risk

Muse API cost and latency. Log usage per task (`ApiCompleter.usage`).
