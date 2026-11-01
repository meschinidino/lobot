# FD report: fluency-aware dose choice

Spec: [specs/FD-fluency-dose.md](../specs/FD-fluency-dose.md).

## Results first

- **Gate: PASS.** Both new frozen doses have final-split fluency above
  0.5: pirate 0.948, cowboy 0.980.
- **Pirate changed from a failing broken patch to a passing working
  patch.** The old dose (L17, 1.5) wrote "ol' ol' ol' ...". Its turn-up
  card failed, because a shuffled control direction at that dose also
  wrote gibberish that the judge scored as pirate (control CI
  [0.378, 0.575]). The new dose (L16, 0.5) writes normal text with a
  light pirate tone. Its card passes: trait +0.333 [0.187, 0.484],
  control [-0.033, 0.011]. Turn-up is now eligible.
- **The trait gain is smaller.** Pirate's turn-up gain fell from +0.780
  (broken text) to +0.333 (fluent text). A plain prompt still gives more
  (+0.910 at a cost of 0.205, Pareto verdict `tradeoff`).
- **Cowboy did not change.** The rule kept the same dose (L17, 0.5).
  Its card still fails the minimum trait mean (+0.179).
- **The NLL check now blocks fluent doses.** Every dose 1.0 cell was
  fluent for cowboy (0.75-0.85), but its NLL rise was over 1.0. FD keeps
  the NLL check by design (one change at a time). See "Next".

## What changed

- `trait_eval.coherent_doses`, `trait_eval.best_dose`: the FD rule
  (`DOSE_RULE = "nll+fluency-v1"`, `NLL_MAX_RISE = 1.0`,
  `FLUENCY_FLOOR = 0.5`).
- `tune_layer_dose` scores every tuning cell with
  `pareto.FLUENCY_LABEL` and stores `fluency` in the table.
- `report.json` `tuning.rule` records the rule.
- `tests/test_dose_rule.py`: 6 tests (fluent dose wins, low-NLL broken
  dose rejected, high-NLL dose rejected, inclusive limits, no coherent
  dose, fluency stored in the tuning table).
- Full suite: 677 tests pass, 5 skipped (fakes only, no weights).

## Live results

Llama 3.2 3B, llama lane (f16), full schedule, seed 0, `--pareto`.
Runs: `logs/fd_pirate_llama` (621 s), `logs/fd_cowboy_llama` (797 s).
Tuning cells are (trait judge, NLL, fluency); base NLL 0.232, base
fluency 0.960. `*` marks the chosen cell.

Pirate tuning:

| layer | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|
| 17 | 0.234, 0.622, 0.956 | 0.874, 1.323, 0.196 | 0.864, 1.090, 0.018 | 0.771, 1.422, 0.020 |
| 16 | **0.294, 0.749, 0.954 \*** | 0.951, 2.044, 0.557 | 0.855, 2.776, 0.087 | 0.755, 1.226, 0.018 |
| 20 | 0.090, 0.354, 0.994 | 0.839, 2.194, 0.341 | 0.756, 1.487, 0.021 | 0.713, 1.128, 0.007 |

L17 at 1.5 (the old choice) passes the NLL check (+0.86) with fluency
0.018. FD rejects it.

Cowboy tuning:

| layer | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|
| 20 | 0.124, 0.313, 0.927 | 0.745, 1.408, 0.847 | 0.764, 2.020, 0.098 | 0.286, 1.474, 0.003 |
| 19 | 0.147, 0.335, 0.928 | 0.678, 1.467, 0.783 | 0.836, 1.703, 0.017 | 0.725, 1.608, 0.006 |
| 17 | **0.245, 0.481, 0.921 \*** | 0.813, 1.622, 0.749 | 0.905, 1.540, 0.034 | 0.802, 1.398, 0.025 |

Final split (n=30), before and after:

| patch | layer, dose | turn-up gain [CI] | control [CI] | card | Pareto fluency |
|---|---|---|---|---|---|
| pirate, old | 17, 1.5 | +0.780 [0.737, 0.819] | [0.378, 0.575] | FAIL (control) | 0.017 |
| pirate, FD | 16, 0.5 | +0.333 [0.187, 0.484] | [-0.033, 0.011] | **PASS** | **0.948** |
| cowboy, old | 17, 0.5 | +0.259 [0.137, 0.391] | [-0.027, 0.050] | FAIL (mean) | 0.979 |
| cowboy, FD | 17, 0.5 | +0.179 [0.074, 0.296] | [-0.045, 0.022] | FAIL (mean) | **0.980** |

Cowboy's tuning gain moved from 0.146 to 0.102 with the same layer and
dose. The llama lane batches 32 sequences, so greedy replies are not
bit-identical between runs.

Pareto (pp-v2) after FD: both verdicts stay `tradeoff` against
`prompt:label`. Pirate's best vector point (vector@0.5, L16) minus the
prompt: trait -0.577 [-0.723, -0.428], cost -0.171 [-0.228, -0.105].

Guard (screen record, not enforced): pirate FD flags refusal on 1 of 8
prompts (old: 5 of 8). Cowboy FD is clean under the privacy v2 rule
(8 leaks, controls 13, 5, 11).

## Acceptance

1. Unit suite passes; the rule has the four required tests. **PASS.**
2. Live: both new frozen doses have final-split fluency at least 0.5.
   **PASS** (0.948 and 0.980).

## Next

- The NLL check rejects every fluent dose above 0.5 here. A rule with
  fluency only (or relevance plus fluency) would pick dose 1.0 for
  cowboy (tuning fluency 0.75-0.85, trait 0.68-0.81). That change needs
  its own spec and a fixed rule before it runs, because these tables
  are now seen.
- The judge is the local 3B model. Fluency is a judge score, not a
  human rating.
