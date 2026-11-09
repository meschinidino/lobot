# Metrics: what "faster" and "better" mean

Warning first: no single number proves a run is faster or better.
A run can cut seconds by doing less work. A run can raise a judge
score by pleasing the judge. This page defines the small set of
numbers that resist those moves, and the numbers we do not use.

Every Create report carries two blocks:

- `provenance`: what the run was. Git sha (with `-dirty` on an
  uncommitted tree), bank version and `bank_sha256`, resolved
  schedule params and `gen_tokens`, judge label hash, split
  sizes, tuning candidates tried, finals recorded.
- `metrics` (`metrics-v1`): what the run did. Per-verb paired
  effects, judge-independent side effects, time accounting,
  physical work counters, and comparability flags.

## Better results

The quality headline is the paired trait-minus-control effect per
verb: mean and lower 95% interval of per-prompt differences
against the shuffled-direction control, on the full 30-prompt
final split. It travels with:

- Win rate and worst decile of the same differences. Means alone
  hide tails.
- Side-effect diagnostics on the same final pairs, measured
  without the judge: refusal rate base vs steered, empty rate,
  repetition rate, realized reply length in words. These are
  text heuristics. They catch refusal, emptiness, and loops;
  they do not prove a reply is relevant or coherent.
- The winner's-curse gap: tuning gain minus final mean. A large
  gap says tuning overfit.
- Vector-minus-prompt paired effect at the same n. The two
  controls differ (shuffled direction vs neutral instruction),
  so compare effects, not pass rates.

A quality headline is eligible only when `comparability` has no
quality reasons: full schedule, fixed n of 30, no early stop, no
simulated port (fake judge, noop guard, fake dose rule), guard
enforced, complete provenance. Fast runs are a screen. Their
numbers never sit beside full numbers.

## Faster

The speed headline is seconds at fixed work, read from
`scripts/compare_runs.py`. The script prints a speed delta only
when both runs are speed-eligible, their lane work counters
match (generations, judge pairs, NLL pairs), and the paired
effect did not regress. Judge and NLL counters count lane
requests; the `*_unique_pairs` counters count distinct executed
evaluations after memoization. Seconds per 1k generated tokens
is a labeled diagnostic only: it rewards verbosity.

Speed numbers cover Create compute stages only. Model load, GPU
queue wait, and follow-up jobs are not in `report.seconds`. Do
not quote it as request-to-result time. A warm run
(`capture_cache_hit: true`) is not speed-eligible: a cache hit
does less work.

## Compare two runs

```sh
uv run python scripts/compare_runs.py OLD/report.json NEW/report.json
```

Exit 2 means no winner: the runs differ in a comparison-key
field (schedule, gen_tokens, bank hash, judge label hash, model,
revision, lane, judge, seed), or one run is headline-ineligible.
The script names the field or the reason and stops.

## Numbers we do not use as headlines

- Raw `seconds` or stage times alone. Cache hits, a smaller
  schedule, and early stop all cut them by doing less.
- `tok_per_s` from a single generation. It includes prefill and
  shrinks when a reply ends early.
- Tuning gain, extraction separability, read-gate agreement.
  All three are selection-set numbers.
- Pass bits and knob rate. They hide effect size and tails.
- Mean fluency, NLL, and the Pareto harmonic mean. Means trade
  one failure against another; NLL stays low on repeated text.
- Pareto `best_vector` gain. It is selected on the final split,
  so it is an oracle number, not a selectable result.
- Reverify on the same final prompts with the same judge. It
  proves the artifact replays, not that the effect is real.
- Guard "pass" from a run where the guard was not enforced.

## Known gaps

These are open work, tracked in `docs/TODOS.md`. Until they land,
treat quality headlines as within-bank, within-judge claims:

- The prompt bank has 120 prompts, so the reserve split is
  empty. No untouched test bank exists yet.
- The final-split ledger lives in one process. Seed shopping
  across runs is visible in job records but not yet blocked.
- One judge family scores effect and fluency. The side-effect
  block avoids that judge, but the paired effect does not.
- Load and queue time are unmeasured. No request-to-artifact
  latency number exists yet.
