# METRICS night run, 2026-11-08/09

Warnings first:

- Run B (first speed fix) was slower than baseline. Its outputs
  were byte-identical, so the regression is machine state, not
  work. No A-vs-B speed claim stands.
- Run C is faster and earned a knob, but it froze a different
  layer and dose than run A. Speed and quality changed together.
  Read the table as one uncontrolled comparison, not a proof.
- The prompt baseline beat the vector on every pirate run and
  on cheerful. Steering is not yet better than prompting on
  these seeds. The metrics now say this in every report.

## What ran

All runs: Llama-3.2-3B-Instruct, seed 7, full schedule, guard
enforced, no Pareto. Protocol was written before run A.

| Run | Code | Seconds | Frozen | update_up paired [95%] | Verdict |
|---|---|---|---|---|---|
| A pirate | sequential lane | 2212.6 | L10 @ 1.0 | 0.819 [0.773, 0.861] | blocked |
| B pirate | A + prefill timer | 3704.5 | L10 @ 1.0 | 0.819 (identical) | blocked |
| C pirate | B + batched lane | 1372.8 | L13 @ 0.5 | 0.447 [0.332, 0.561] | knob |
| C cheerful | batched lane | 1339.1 | L15 @ 0.5 | 0.352 [0.242, 0.461] | blocked |

Stage split, run A: tune 47%, baseline 29%, final 16%,
read gate 7%. Work counters: 880 generations, 51,032 new
tokens, 1,720 judge requests (1,192 unique).

## What the numbers support

1. Generation dominated Create. Sequential generation was most
   of run A. The interleaved bench measured batched generation
   at 8.8s against 69.8s sequential for 30 prompts.
2. Run C finished in 0.62x run A's compute seconds with equal
   generation, judge, and NLL request counts. Confounds remain:
   machine heat, sleep in run A, a different frozen edit, and
   guard work outside the lane counters (4s blocked path against
   100s full screens). Codex's wording stands: "C recorded 1.61x
   lower compute time than A in one uncontrolled comparison,
   with a different selected edit."
3. The paired-effect drop has a named cause. Batched replies
   changed 5 of 30 fluency scores at L10 @ 1.0. Its fluency p10
   moved 0.338 to 0.297, across the 0.30 dose gate, so tuning
   picked L13 @ 0.5. This is threshold fragility in dose
   selection. It is not yet evidence that batching weakens a
   fixed edit.
4. The old double prefill cost +5.5% per generation. Run B's
   large slowdown does not come from the replacement timer
   (+2.5% interleaved). KV-cache prefill reuse changed token
   ids and was rejected.

## Bugs found and fixed tonight

- Two eligible verbs crashed the verdict stage (ledger key).
- A down-only knob shipped the up report card and exported the
  opposite edit. `patch.signed_dose` now carries the sign.
- Short lane outputs silently shrank score denominators.
- Removing the extra prefill shifted chat dial taps by one.
  `dial_for_index` now reads the producing forward.

## Next measurement (Codex, agreed)

One counterbalanced crossover, no retuning: both frozen edits
(L10 @ 1.0 and L13 @ 0.5), each run sequential and batched on
one fixed prompt set, prefill measurement off, fallback off.
It separates execution mode from edit choice in one sitting.

## Artifacts

- `logs/metrics_pirate_a`, `logs/metrics_pirate_b`,
  `logs/metrics_pirate_c`, `logs/metrics_cheerful_c`
- App jobs: `metrics-pirate-c`, `metrics-cheerful-c`
- Benches: `/tmp/prefill_bench.py`, `/tmp/prefill_bench2.py`,
  `/tmp/prefill_bench3.py`; readouts `/tmp/read_metrics.py`,
  `/tmp/ab_identity.py`
- Tests: 925 unit tests pass, 6 skipped. Real weights ran only
  in the four Creates and three benches above.
