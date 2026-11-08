# SH: Ship to 10 — report

Spec: `docs/roadmap/specs/SH-ship-to-10.md`.
The bars did not move: facet judge, gain 0.30, fluency worst
decile 0.30. This report says what passed, what failed, and
what still needs a person.

## Phase A: land the code

1. PR #61 reviewed in slices (guard batching, WHY NO KNOB,
   ML steps 1-2, GPU stack guard, learned vectors, learned in
   app, facet judge, few-shot training). Run-output files
   (`store/traits.json`, calibration) read line by line.
   Findings reported before any fix, in the session.
2. Review blockers fixed on `feature/ml-combine-and-guards`
   and pushed (`7945212`). Gate on the branch:

   ```
   Ran 871 tests in 28.138s
   OK (skipped=6)
   PYFLAKES_CLEAN
   ```

   The fixes: the app serves the confirmed learned scale and
   re-checks both bars plus the full facet list at knob time;
   the GPU stack guard covers the agent and transfer CLIs (a
   test scans every daemon call site); WHY NO KNOB states the
   tested scope; vector loads use `weights_only=True` and
   validate shape and values; confirms record the spent prompt
   set by hash; agent-made labels claim no hand validation in
   `store/traits.json`; a job with no readable report earns no
   knob; a short engine batch fails loud.
3. The newest `develop` did not re-break the work.
   `update-index --refresh` ran before the diff check.
4. PR #61 merged to `develop` (`dc949bd`). Gate on the merged
   tree:

   ```
   Ran 873 tests in 33s (full suite)
   OK (skipped=6)
   PYFLAKES_CLEAN
   ```

   `develop` merged to `main` by PR #63 (`d1fd5b5`). The
   merged `main` tree and `develop` tree were identical for
   this scope at merge time.

## Phase B: hard behavior, bigger confirm sets

Model lane: `SH-B5-RETRAIN90`, `SH-B6-DIAG`,
`SH-B6-CAPPED-CONFIRM`, `SH-B79-CONFIRMS` (start and end
lines in `model-lane.md`). One job at a time.

5. Training questions grew from 30 to 90 (`--extra-train`,
   disjoint-checked against the fresh set). Scale search on
   tuning picked 1.0. Confirm on fresh_questions_v14 (50
   brand-new questions):

   ```
   BEST SCALE 1.0 CONFIRM {'gain': 0.848, 'fluency_mean': 0.588,
   'fluency_min': 0.004, 'fluency_p10': 0.202} FAIL
   ```

6. Per-prompt dose cap: a collapsed reply retries at half
   dose (most-common-word share over 0.35). Confirm on
   fresh_questions_v15 (50 brand-new questions):

   ```
   FIXED SCALE 1.0 CONFIRM {'gain': 0.840, 'fluency_mean': 0.593,
   'fluency_min': 0.048, 'fluency_p10': 0.154, 'capped': 0} FAIL
   ```

   The cap fired zero times. Diagnosis on the spent v15 set:
   the 8 replies under fluency 0.30 all have gain over 0.70
   and read as coherent English. Repetition is at most 0.084.
   Length does not track the score (correlation 0.10).
   Inference, not measured fact: the local fluency judge scores
   hostile text as not fluent. No clean input-side signal
   separates the tail.
7. Rude verdict: **no knob**. The family ships as an honest
   no-knob on its evidence. Its mix stays in
   `learned_vectors.pt`; `_learned_pass` returns nothing for
   it, so the app offers no learned knob.
8. Full guard check on the rude mix at scale 1.0: BLOCKED.
   Refusal screen, min ratio 0.004. The verdict stands in
   `learned.json` (`status: full_guard_run_sh_b8`). The mix
   runs Research-only, on purpose.
9. Shipped knobs at 50 brand-new questions, one set and one
   verdict each (bars: gain 0.30, fluency worst decile 0.30):

   | Behavior | Gain | Fluency mean | Fluency p10 | Verdict |
   |---|---|---|---|---|
   | Catholic (L10, dose 1.0) | 0.565 | 0.691 | 0.324 | PASS |
   | Pirate (L16, dose 0.5) | 0.225 | 0.974 | 0.921 | FAIL |

   Pirate text stays clean. Its gain falls under the bar at
   50 questions (35 of 50 replies gain under 0.1). Its old
   30-question card said 0.329. No re-roll ran. This is a
   failed gate with numbers, not a pass.

## Phase C: gaps and hosting (branch `feature/SH-phase-b`)

10. Learned mixes show a visible no-dial state. The read menu
    lists them with "no dial for learned mixes". Chat and run
    answer "No read dial for learned mixes: '<trait>' steers
    many layers at once." The old false "Create the trait
    first" message survives only for traits that truly do not
    exist.
11. `kind: "multi"` patches publish, install, and re-verify.
    The registry reads the learned confirm card and refuses
    mixes under the fixed bars. Multi loads reject duplicate
    layers, unlisted tensors, dtype or shape drift, zero norm,
    and non-finite values. The llama.cpp export raises one
    clear error for multi patches. Round-trip tests:
    `tests/test_patch_registry.py` (MultiPatchTest).
12. Hosting assets: `GET /health`, a "Run it on a server"
    section in the README (one process, one GPU slot, about
    24 GB, the start command, the health check command). Live
    check in replay mode on this Mac:

    ```
    {"ok":true,"model":"Qwen/Qwen3.5-4B","warm":false,"backend":"torch"}
    research_allowed False model Qwen/Qwen3.5-4B
    ```

    This proves the start command and the health route on the
    Mac in replay mode (no weights). A later check (SH-LIVE-CHECK,
    2026-11-08) ran the merged server live on this Mac with
    Llama weights: /health answered warm: true, the Catholic
    knob steered a reply on real weights, and pirate answered
    404 after its takedown. It does not prove a run off the
    Mac. Open: one live run on a hosted machine. No host with
    the owner's account is set up yet; `deploy/space` carries
    the read-only replay demo, not live weights.
13. Research mode is an owner switch:
    `LOBOT_ALLOW_RESEARCH=1` at server start. Every router
    takes the client flag AND the owner switch. The Research
    button hides when the server does not allow it. Every
    Research run writes an audit record (`research_run` with
    `would_block`), tested against a temp audit log. Tests
    never write to `~/.lobot/audit.jsonl`.

## Gate

```
Ran 890 tests in 30.328s
OK (skipped=6)
PYFLAKES_CLEAN
```

The suite ran with `LOBOT_AUDIT_LOG` pointed at a temp file,
on fakes, no server running. Live checks ran on real weights
through the model lane; the numbers above name each mode.

## Success criteria from the spec

| # | Criterion | State |
|---|---|---|
| 1 | PR #61 merged to develop and main | Done (dc949bd, d1fd5b5) |
| 2 | No silent gap: multi reads, registers, shows its state | Done: read shows no-dial, registry and re-verify read multi |
| 3 | Every shipped behavior passes on 40+ fresh prompts | Catholic PASS (50). Pirate FAIL (50, gain 0.225). Rude no-knob, guard-blocked |
| 4 | Rude knob passes fresh at 40+ with strong text, or no-knob verdict names the numbers | Done as no-knob: 0.848 / 0.202 and 0.840 / 0.154 at 50 fresh. Its evidence says so |
| 5 | Full test suite passes, lint clean | Done: 890 tests OK, pyflakes clean |
| 6 | App runs at one address off the local machine | Open: health route and start command proved on the Mac (replay). Needs a host with the owner's account |

## Known gaps

- Hosting off the Mac waits for a host choice (spec risk 3).
  The repo carries a replay demo under `deploy/space`, not a
  live-weight host.
- The agent still sets `hand_validated` when it registers a
  family from its own labels. The committed store records the
  label source and claims no hand validation. A field rename
  waits for a contract change.
- Pirate failed its 50-question re-confirm. On Dino's order
  (2026-11-08) it is taken down: `jobs/pirate-fv3/taken_down.json`
  records why. The app offers no pirate knob; the evidence
  stays readable. The takedown is a job record, so the newest
  job's takedown wins over every older pirate knob.
