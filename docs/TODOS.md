# TODOs

Deferred work, each with its reason. The ordered plan is in
[roadmap/README.md](roadmap/README.md).

## Open

### Decide chat cache semantics (O6, open decision)
- Question: when edits are active, do they apply only to new tokens (clean
  prompt prefill), or from the prompt on?
- Now (fact, not a decision): both backends apply edits from the prompt on.
  The torch hooks stay active during prefill; the llama daemon sets the cvec
  before prefill and clears its prefix cache when the cvec changes.
- Choosing "new tokens only" needs a clean prefill, edits attached at decode
  step 0, and a re-run of the D3 equality gate. Details:
  [archive/open-questions.md](archive/open-questions.md).

### Create in about two minutes (C2 gate)
- Now: the llama lane runs a full Create in 480 s (torch 2043 s) with the
  same outcome. The gate is 120 s.
- Left: judge and NLL scoring are prefill-bound; ablation generations and
  activation capture stay on torch.

### Live check of judged install verify (IN)
- `judged_n` is built and unit-tested. No `llama-cli` or `ollama` binary
  is on the dev Mac, so it never ran live.

### A live "model update breaks a patch" case (behavior CI)
- CI ran live; the published patch held on torch and Q4_K_M. Showing a
  failure needs a patch with a narrow interval or a further-off build.

### Fairer privacy rule
- Random directions leak 2-16 of 40 planted secrets, so "2 above the worst
  of 3 controls" is noisy. A rule on the control mean and spread needs more
  controls (each costs about 40 generations).

### Dose rule without the NLL check (FD finding, 2026-11-01)
- Planned as JS ([specs/JS-judged-scores.md](roadmap/specs/JS-judged-scores.md)).

### GR v3 on a fresh guard set (2026-11-01)
- Planned as GH ([specs/GH-guard-holdout-repair-v3.md](roadmap/specs/GH-guard-holdout-repair-v3.md)).

### Make the model choice persistent and default to Llama (2026-10-30)
- The hero model is Llama 3.2 3B, but `steer.MODEL_ID` (Qwen) is still the
  startup default. Remember the last model picked, and start on Llama when
  its weights are present. Owner: UI with a small server change.

### Scale to 70B-class models (1000X SC)
- 70B weights need about 40 GB even at 4 bits; the dev Mac has 24 GB.
  Needs rented GPUs. The registry, audit and verify paths do not depend on
  model size.

### Bake Delete into model weights (deferred 2026-10-29)
- Run-time Delete works only inside lobot. Baking it in means editing the
  weights against the trait direction and exporting a full model (GBs).
- Why deferred: large effort, multi-GB outputs, and it is the direct path to
  distributing models with safety refusals removed.
- Must keep: the guard and its record in every exported artifact.

## Done

- **Open-ended Create** (deferred 2026-10-29): built as C3 open
  descriptions; noir and cowboy passed; the Muse Spark agent automates it.
