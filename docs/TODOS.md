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
- FD added a fluency floor to Create's dose choice (reports/FD.md).
  Pirate now passes with fluent text. But the NLL check rejects every
  dose 1.0 cell for cowboy, though those replies are fluent (0.75-0.85).
- Fix: a rule on fluency (and maybe relevance) without the NLL rise.
  Write the rule in a spec before it runs; the FD tables are seen.

### GR v3 on a fresh guard set (2026-11-01)
- Gradient repair cut pirate refusal flips 7 -> 1 with the trait kept,
  but 1 flip still flags. Re-linearizing at the repaired point would
  likely close it, but the guard prompts were already used as the
  test twice.
- Needs: a fresh held-out refusal and honesty set, then v3
  (iterated gradient projection) evaluated once on it.

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
