## Goal

Take lobot from a working research build on one Mac to a shipped
tool: merged code, one hard behavior passing honestly, hardened
margins, closed learned-mix gaps, and a host other people can run.

## Success Criteria

1. PR #61 is reviewed and merged; `develop` reaches `main`.
2. The rude family earns a knob, or ships with an honest no-knob
   verdict and the reason on its evidence page.
3. Every shipped behavior passes on at least 40 fresh prompts.
4. Learned mixes have a read dial or a visible "no dial" state,
   and registry/re-verify read multi-layer patches.
5. The app runs on a host that is not Dino's Mac, with Research
   mode behind an explicit, recorded choice.

## Approach

Ship in three phases. Phase A makes the current code landable.
Phase B makes the hard behavior pass or fail honestly at bigger
sample sizes. Phase C closes product and ops gaps. No phase
starts before the prior phase gate passes.

Key decisions:

- Confirm sets grow to 40-60 prompts. The 20-prompt sets swung
  gain by ±0.1 and flipped verdicts twice; bigger sets end that.
- The facet judge and both gates (gain 0.30, fluency worst
  decile 0.30) stay as they are. We fix inputs, not bars.
- Research mode stays. It gets an owner-only switch and an audit
  record per run, not removal.

## Steps

### Phase A: land the code (items 1, 6)

1. Review PR #61 in slices: guard batching, WHY NO KNOB, ML
   steps 1-2, GPU stack guard, learned vectors, learned-in-app,
   facet judge, few-shot training. Check the run-output files
   (`store/traits.json`, calibration data) line by line.
2. Fix review findings on `feature/ml-combine-and-guards`. Run
   the full suite and `uvx pyflakes src scripts` after each fix.
3. Merge to `develop`. Run the full suite on the merged tree.
4. Merge `develop` to `main` by PR at the milestone.

### Phase B: pass or honestly fail the hard behavior (items 2-5)

5. Grow training questions for learned vectors from 30 to 90,
   same tuning/fresh split discipline. Retrain with `--shots`.
6. Add a per-prompt dose cap only if step 5 still fails the
   fluency tail: hard prompts get a smaller push, easy prompts
   keep the full push. Gate the cap with a unit test.
7. Write confirm sets of 40-60 fresh questions per behavior that
   ships (rude, Catholic, pirate). Confirm at the chosen scale.
   One set, one verdict, no re-rolls.
8. Run the full guard check on each passing mix. Record the
   verdict in `learned.json` and in the patch file. Expect the
   rude mix to stay Research-only.
9. Re-confirm Catholic and pirate at the bigger set size, so all
   shipped behaviors carry the same evidence strength.

### Phase C: close gaps and host it (items 7-10)

10. Read dial for learned mixes: read each layer's vector on its
    own, show a band per layer, or show "no dial for learned
    mixes" in the UI. Pick one; the silent gap is the defect.
11. Teach registry and re-verify to load `kind: "multi"`
    patches, or block them with a clear error. Add round-trip
    tests next to `tests/test_patch_registry.py`.
12. Host the app off the Mac: one process, one GPU slot, a start
    command in the README, a health route check. The session
    server stays the dev path.
13. Research mode: an owner-only toggle, off by default, every
    run recorded in the audit log with `would_block`. Write the
    rule in `docs/guides/app.md`.

## Validation Plan

- Phase A: `uv run python -m unittest discover -s tests -t .`
  and `uvx pyflakes src scripts`, on the branch and again on
  merged `develop`. Both must pass.
- Phase B: each confirm writes its set name, gain, fluency mean
  and worst decile to `learned.json`. The report in
  `docs/roadmap/reports/ML.md` quotes them. A fail is a result:
  it ships as a no-knob verdict, not a retry loop.
- Phase C: registry round-trip tests for multi patches; a live
  check that the hosted app answers on its real address; a
  Research run that shows `would_block` in the audit log.
- Every model run takes the model lane and logs start and end
  in `docs/roadmap/model-lane.md`.

## Risks / Open Questions

- The fluency tail may survive steps 5-6. Fallback: the rude
  family ships as an honest no-knob. The tool ships; that
  behavior does not.
- Bigger confirm sets cost model time: roughly 3x per behavior.
  One lane, one job at a time.
- Hosting needs a machine with a GPU. Which host is open; the
  plan works for any always-on box Dino controls.
- None open on gates or bars: both stay fixed.
