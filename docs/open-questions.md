# Open questions (backend waves 0+1)

## O6 — chat cache semantics (pending, owner: wave-2 lead)

Plan (`docs/plans/lobot-10x-plan.md`, rows 114/485): pending,
non-blocking, "decide before the wave-0 performance gate".

Question: when edits are active, do they apply to **future tokens
only** (prompt prefill stays clean; default in the plan) or does the
context get **recomputed** under the edits?

What it blocks: the wave-0 performance gate (batched filmstrip memory
+ time on 24 GB) is ordered "with chat cache semantics decided
first". Numbers can still be measured on the current implementation,
but the gate cannot be read as a ship decision until O6 is answered.

Current implementation (fact, not a decision): hooks register before
`generate()` and stay active for the prefill forward pass and every
decode step, so steered runs prefill under the edits. Changing this
to future-tokens-only would need a clean prefill + hook attach before
decode (and a D3 re-proof, since token ids may shift).

Options for the owner:

1. Future tokens only (plan default): prefill clean, edits attach at
   decode step 0. Cheaper to reason about; needs the D3 gate re-run.
2. Recompute context: keep edits active from prefill (current
   behavior). No code change; bless the measured numbers.

No other open decisions. Everything else on the wave-0/1 path is
approved in the plan ledger (D1–D14, O1–O5, O7, O8, S3-1).
