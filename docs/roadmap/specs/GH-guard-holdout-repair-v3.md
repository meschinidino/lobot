# GH Fresh guard set and guard repair v3

Owner: Muse (coding agent). Lane: no-model for code, sets and tests;
one model slot for the live runs. Depends on: GR, FD. Time: two days.

## Warning

The current refusal and honesty guard prompts were used to choose GR
v1 and v2. They are no longer a clean test. Any new repair method must
be judged on a set that nobody has used.

## Goal

1. Split the guard into a **dev** set (the current prompts) and a new
   **test** set (fresh prompts, written now, used once per method).
2. Build repair v3 and judge it on the test set.

## Part 1: test set

1. Write 16 new refusal prompts and 16 new honesty (false-premise)
   prompts. Cover the same kinds of risk as the dev set, with new
   wording and new topics. No prompt may share 5 or more consecutive
   words with any dev or fit prompt (check in a test).
2. Write 40 new privacy items (planted secret + bait question) the
   same way.
3. Commit them in `guard_screens.py` as `*_TEST_V1`. The current sets
   become `*_DEV`. Keep the old names as aliases so nothing breaks.
4. `Guard` gains `split="dev" | "test"`. Publishing (registry) uses
   `test` from now on. The verdict records which split it used.
5. Log every use of the test set in `docs/roadmap/guard-test-log.md`:
   date, method, patch, result. A method gets one look per patch.

## Part 2: repair v3 (iterated gradient projection)

GR v2 removed the first-order safety cost at the original direction.
After the projection, the model sits at a new point, and new gradient
rows appear. v3 repeats the step:

1. `unit_0` = the patch direction.
2. For k = 1..3: take the gradient rows at `unit_{k-1}` (same targets
   and fit prompts as v2). Add them to the row set. Project the
   original direction off the span of all rows so far. Normalize to
   get `unit_k`.
3. Stop early when the dev-set screen is clean.

Candidates, fixed now: `gradient-iter` at 1.0x and 0.75x dose, then
v2 `gradient` at 0.75x (the GR best). The choice rule is unchanged:
the first candidate clean on the **dev** screens with a positive tuning
gain. Then the chosen candidate gets, once:
- the full confirmed guard on the **test** set,
- the turn-up report card on the final split,
- a Pareto row at its dose (PP).

## Patches to test

1. Pirate, old patch (L17, dose 1.5; refusal 7, honesty 5 on dev).
2. Pirate, FD patch (L16, dose 0.5; refusal 1 of 8 on the dev screen).
3. Any other knob in `jobs/` or `logs/` whose confirmed guard flags,
   listed before the run.

Noir is excluded: its trait gain is near 0, so repair cannot be judged.

## Files

- `src/model_lobotomy/guard_screens.py`: test sets, split switch.
- `src/model_lobotomy/repair.py`: `GRADIENT_ITER_CANDIDATES`,
  `iterated_rows(...)`.
- `scripts/repair_patch.py --method gradient-iter --split test`.
- Tests: overlap check (5-word shingles), split switch, iteration with
  fake gradients (rows grow, result orthogonal to every row).

## Acceptance (fixed now)

1. Unit suite passes.
2. Gate: at least one tested patch reaches `repaired`, with a clean
   full guard on the **test** set and a passing card.
3. The report gives every candidate's dev and test numbers, pass or
   fail, and the Pareto row of the chosen candidate.

## Known risk

Random directions also leak secrets (privacy v2). If a privacy flag is
caused by the size of the push, projection cannot fix it; only a lower
dose can.
