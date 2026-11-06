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

### Changed

- Evidence pages state why no knob exists, with the tested scope,
  instead of a bare NO KNOB verdict.
