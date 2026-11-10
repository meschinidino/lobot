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

## Crossover result (2026-11-09, METRICS-X)

Ran as specced above. Order: L10 sequential, L13 batched,
L10 batched, L13 sequential. One warmup generation first.
Prompt set: the seed-7 final split (30 prompts). Each cell
generated base, steered and shuffled-control replies, then
scored trait, control and steered fluency. Runner:
`/tmp/crossover.py`. Artifact: `logs/metrics_crossover/crossover.json`.

| Edit | Mode | Paired effect [95%] | Fluency p10 | Gen seconds |
|---|---|---|---|---|
| L10 @ 1.0 | sequential | 0.819 [0.773, 0.861] | 0.328 | 169.7 |
| L10 @ 1.0 | batched | 0.815 [0.768, 0.858] | 0.328 | 18.3 |
| L13 @ 0.5 | sequential | 0.447 [0.332, 0.561] | 0.958 | 165.1 |
| L13 @ 0.5 | batched | 0.447 [0.332, 0.561] | 0.958 | 18.7 |

What this proves, in one sitting, at fixed edit:

1. Batching does not change quality at a fixed edit. The
   paired-effect move is -0.004 at L10 and 0.000 at L13.
   Fluency p10 is identical in both modes at both edits.
   Steered texts are identical in 58 of 60 prompts. The two
   L10 text diffs move the trait score by 0.002 and -0.005.
2. The run A vs run C gap is edit choice, not execution
   mode. L10 @ 1.0 scores 0.819 paired; L13 @ 0.5 scores
   0.447 paired in both modes. The 0.37 gap travels with
   the edit.
3. Batched generation is about 9x faster at equal work
   (165-170s vs 18-19s for 90 generations per cell). Cell
   totals, judge included, run 4.5-5.5x faster. The judge
   still scores one pair at a time in both modes.
4. The runner reproduces the original finals. L10
   sequential matches run A's 0.819 [0.773, 0.861]. L13
   batched matches run C's 0.447 [0.332, 0.561].

What this does not prove:

- Why tuning picked L13 in run C. This crossover did not
  rerun tuning. At the final split, L10 fluency p10 is
  0.328 in both modes, only 0.028 over the 0.30 gate.
  Small tuning-split jitter can still cross that gate.
  The dose gate stays threshold-fragile; the fragile step
  is selection, not final scoring.
- Any claim past this prompt set, this seed, or this
  sitting. The final split is spent. The timing is one
  counterbalanced sitting, not the repeated-timing
  protocol in `docs/TODOS.md`.

Decision: keep the batched lane. The next open item is
dose-selection stability, not batching.

## Dose-selection fix and payoff run (2026-11-09, METRICS-D)

Root cause, from the stored tuning tables: in run C,
L10 @ 1.0 tuning fluency p10 read 0.297 against the 0.30
floor. One batched reply moved the third-lowest fluency
from 0.338 (run A) to 0.297, and the v3 gate rejected the
strong dose over 0.003. Tuning then froze L13 @ 0.5.

Fix: `fluency-first-v4` is the new default dose rule. The
tail gate reads the worst-decile mean of the per-prompt
fluency scores (floor 0.20) instead of the single p10
value. On the same tuning data the tail mean moves 0.271
to 0.252 between modes, so the dose stays coherent. A
looping worst decile still reads near zero and fails.
Tuning tables now store `fluency_tail_mean`. The
final-split and registry p10 gates do not change. On the
four stored tuning tables, v4 changes one choice: run C's
table now picks L10 @ 1.0, the run A choice.

Payoff run: one full Create, pirate seed 7, guard on,
batched lane, v4 rule, cold capture cache
(`logs/metrics_pirate_d`).

| | Run A (v3, sequential) | Run D (v4, batched) |
|---|---|---|
| Frozen | L10 @ 1.0, gain 0.848 | L10 @ 1.0, gain 0.848 |
| Final paired [95%] | 0.819 [0.773, 0.861] | 0.815 [0.768, 0.858] |
| Seconds | 2212.6 | 562.4 |
| Work (gen/score/nll) | 880 / 1720 / 900 | 880 / 1720 / 900 |
| Verdict | blocked (guard) | blocked (guard) |

Read it with care:

1. The dose fix works in a live tuning run, not only on
   stored tables. Run D froze the strong edit that run C
   lost, at the same tuning gain as run A.
2. Quality held. The paired effect matches run A and the
   crossover batched cell (0.815).
3. No knob ships. The guard blocks L10 @ 1.0 in both runs
   (min_ratio 0.28, same two prompts). A guard block is a
   result: strong pirate at this dose weakens refusal.
   Run C's knob existed only because the weak edit passed
   the guard.
4. The 3.9x total-time ratio is one uncontrolled
   comparison. Work counters match, but machine state
   differs between sittings (run C, same lane, took
   1372.8s). `compare_runs.py` refuses the A-vs-D headline
   (exit 2, guard_blocked on both). The controlled speed
   evidence stays the crossover: about 9x on generation
   at fixed edit.

Tests: 927 unit tests pass, 6 skipped. The two new v4
tests cover the p10-jitter case and the collapsed-tail
case.

## Artifacts

- `logs/metrics_pirate_a`, `logs/metrics_pirate_b`,
  `logs/metrics_pirate_c`, `logs/metrics_cheerful_c`
- App jobs: `metrics-pirate-c`, `metrics-cheerful-c`
- Benches: `/tmp/prefill_bench.py`, `/tmp/prefill_bench2.py`,
  `/tmp/prefill_bench3.py`; readouts `/tmp/read_metrics.py`,
  `/tmp/ab_identity.py`
- Tests: 925 unit tests pass, 6 skipped. Real weights ran only
  in the four Creates and three benches above.
