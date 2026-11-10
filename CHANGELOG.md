# Changelog

Every change to lobot lands here, newest first, under Unreleased.
Move Unreleased entries into a version section when a version ships.

Keep this file up to date on every change: code, docs, or UI. An
entry is one or two short lines. Say what changed and why it matters
to a user. Write in Simplified Technical English, like the docs.

Deferred work lives in `docs/TODOS.md`, not here. Update that file
in the same change whenever work is added, deferred, or finished.

Format follows Keep a Changelog: Added, Changed, Fixed, Removed.

## Unreleased

### Added
- Objective Create metrics (`metrics-v1`) and a provenance
  envelope in every report: paired trait-minus-control effects,
  judge-independent side effects, time accounting, physical work
  counters, bank and judge-label hashes. `scripts/compare_runs.py`
  refuses a faster/better claim when runs are not comparable.
  See `docs/reference/metrics.md`.
- Create report `judge` block now names the effective judge
  (kind and model) instead of a hard-coded local judge.

### Changed
- Torch generation no longer runs a second full forward pass
  over the prompt only to time prefill. `prefill_s` now measures
  time to first token inside the same generate call, and Create
  skips the measurement it discards.
- Create's torch lane generates prompt lists in one batched
  call (`Engine.run_many`) instead of one call per prompt.
  Measured on this Mac: 30 prompts 69.8s sequential vs 8.8s
  batched, 29/30 token-identical. Metrics counters still count
  prompts, tokens, and judge pairs, so speed claims keep their
  denominator.
- Demo slate for Wednesday measured on real weights: five
  knob jobs on five shared fresh prompts. Pirate fv2,
  cowboy, Devout Catholic, Shakespeare fv3, and sports
  commentator take the slots. The guide is
  `docs/guides/demo-wednesday.md`.
- The METRICS crossover confirms the batched lane at fixed
  edits: paired effect moves -0.004 and 0.000 against the
  sequential lane, fluency p10 unchanged, generation about
  9x faster. The run A vs run C quality gap is edit choice
  (L10@1.0 vs L13@0.5), not batching. See
  `docs/roadmap/reports/METRICS.md`.

### Fixed
- Dose selection no longer flips on one borderline reply. The
  new default rule `fluency-first-v4` gates the tuning tail on
  the worst-decile mean fluency instead of the single p10
  value. On the run C tuning table the old gate rejected
  L10@1.0 over 0.003 (p10 0.297, floor 0.30); the tail mean
  reads 0.252 against a 0.20 floor in both execution modes.
  The registry and final-split p10 gates do not change.
- Verdict stage crashed when two verbs earned eligibility in one
  run: the final-split ledger key omitted the verb. All verb
  finals now record under per-verb keys.
- A down-only knob shipped the update_up report card. The patch
  now carries the earning verb's card, and re-verify replays the
  signed dose at the recipe's token budget.
- Short lane outputs vanished into zip loops and shrank score
  denominators. Generation, judge, and NLL paths now fail the
  run on a length mismatch; the llama judge no longer pads a
  missing score with 0.0.
- SH Phase B: `learn_vectors.py --extra-train` grows the learned
  training pool past the 30-question tuning split. A disjoint
  check refuses training questions that sit in the fresh set.
  `measure_fresh.py` takes a prompts file and records the pass
  verdict at the fixed bars.
- SH Phase B: per-prompt dose cap for learned mixes. A collapsed
  reply retries at half dose. The cap rule ships in the confirm
  record, and the app applies it to learned columns.
- SH Phase C: `GET /health` for hosted runs, and a "Run it on a
  server" section in the README.
- Multi patches publish, install, and re-verify. The registry
  reads the learned confirm card (gain, fluency worst decile,
  fresh-set size) and refuses mixes under the fixed bars. The
  llama.cpp export names its blocker for multi patches instead
  of raising a bare key error.
- Learned mixes show a visible "no dial for learned mixes" state
  in the read menu. Chat and run name the same reason instead of
  claiming the trait does not exist.

### Removed
- The pirate knob. It failed its 50-question confirm
  (gain 0.225, bar 0.30). A `taken_down.json` record in its job
  dir takes the trait down: the app offers no pirate knob and
  the shelf marks it TAKEN DOWN. Its evidence stays readable.

### Changed
- Research mode is an owner switch: `LOBOT_ALLOW_RESEARCH=1` at
  server start. A page request alone no longer turns the guard
  off. Every Research run writes an audit record with its
  `would_block` result. The Research button hides when the
  server does not allow it.

### Fixed
- SH Phase A review fixes. The app now serves the scale the
  learned confirm passed, not the search-best scale. A learned
  pass re-checks gain, fluency mean, worst-decile fluency, the
  fresh-set name, and the full facet list before it earns a knob.
- The GPU stack guard now covers the agent and transfer CLIs.
  Both pass the warm torch size. A new test scans every daemon
  call site for the missing value.
- WHY NO KNOB now states the real scope: candidate layers on
  tuning prompts. It no longer claims a fluency cause the sweep
  record does not store. The tuning grid applies the worst-decile
  gate and stops calling NLL a gate for fluency-first rules.
- `scripts/learn_vectors.py` loads vector files with
  `weights_only=True` and rejects missing, zero, NaN, or wrong-shape
  vectors. It keeps the guard verdict in `learned.json` and records
  the confirmed scale in the confirm block.
- Confirms from the combined search record the spent prompt set
  (size and hash) and the true call count. A re-roll on the same
  set stays visible.
- A job with no readable report earns no knob. Research mode can
  still offer it.
- Families whose labels came from an agent, with no user
  confirmation, no longer claim hand validation in
  `store/traits.json`. The label source stands in the data.

### Changed
- The Create judge now scores each word of a multi-word label and
  takes the lowest score. A condescending reply passed the old
  judge as "rude, hateful and mocking". It no longer does.
- Learned-vector training can show the target style with worked
  examples (`--shots`) and set the drift cost (`--drift-w`). The
  rude mix now reaches real rude text. Its fluency tail still
  fails the gate, so it earns no knob yet.
- Learned passes from the old judge no longer count in the app.
  The rude mix needs a new confirm under the strict judge.

### Added
- Learned mixes now run in the app. A passed learned confirm shows as
  a `LEARNED PASS` knob: dose 1.0 is the tested mix. The `learned.lobot`
  multi-layer patch downloads from the evidence card. A mix the guard
  blocks shows `GUARD` and runs only in Research mode.

- Full-layer sweep evidence for the evil-and-mean family: all 28
  layers at 4 doses (112 tests), with a fresh-prompt check of the
  one clean strong cell (layer 12 at dose 0.5 fell from 0.338 to
  0.129). The evidence page now states this scope in its WHY NO KNOB
  box.
- Spec ML (`docs/roadmap/specs/ML-multi-layer-search.md`): a
  LayerNavigator-style layer score, then a greedy combined-layer
  search with a fresh-prompt confirm.
- This changelog, with a standing rule to update it on every change.
- Layer score (ML step 1): `separability` returns `d_prime`,
  `consistency`, and `steer_score`; Create reports store
  `layer_scores`; the tuning rank uses `steer_score` first.
- Combined-layer search (ML step 2): `layer_search.py` plus
  `scripts/combine_search.py`. On the evil family it found
  L16+L27 at dose 0.5: search gain 0.343, fresh confirm 0.260
  against the 0.30 bar, so no knob, but closer than any single
  layer. Full result in `docs/roadmap/reports/ML.md`.
- Learned steering vectors (ML v3): `learned.py` plus
  `scripts/learn_vectors.py`. One trainable push per layer, model
  frozen, drift cost on. On the evil family the trained mix at
  scale 0.625 is the first method to pass a fresh confirm:
  gain 0.402, mean fluency 0.837, worst-decile 0.392 on 20
  brand-new questions.
- Steering probe score (ML v2): layers rank by clean single-dose
  gain, not read strength. On the evil family the probe ranks
  layer 12 third of 28. The v2 winner (L11 at 0.5) confirms at
  gain 0.572 on 20 brand-new questions but fails the fluency
  tail gate (worst decile 0.026, gate 0.3). Still no knob.

### Fixed

- Agent llama lane crash (`AcceleratorError` with a garbage
  index, Catholic run, Qwen 4B): the daemon start now checks the
  combined GPU stack size and refuses early with a clear message
  when the GGUF plus the warm torch weights pass 14 GB. Stacks
  that fit (Llama 3B, or a q4 daemon) still run. Regression tests
  in `tests/test_llama_stack_guard.py`.
- Lane daemons no longer stay warm after a job. Create jobs and
  agent runs close their llama daemon at the end, so at rest only
  the torch model holds the GPU. One new regression test pins the
  close after a Create job.
- Agent run page now names the model and lane the run itself used.
  Before, a run that failed on Qwen sat next to a sidebar switched
  to Llama and read as a model-selection bug. The stack guard
  message also names the model now.

### Removed

- `CLAUDE.md` at the repo root. Its one rule that `AGENTS.md` did
  not cover (flag UI code that breaks `docs/DESIGN.md` in review
  or QA) now lives in `AGENTS.md`. One rules file remains.

### Changed

- Evidence pages state why no knob exists, with the tested scope,
  instead of a bare NO KNOB verdict.
