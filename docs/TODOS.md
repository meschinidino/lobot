# TODOs

Deferred work, each with its reason. The ordered plan is in
[roadmap/README.md](roadmap/README.md).

Keep this file up to date on every change. When work is deferred,
add it here with its reason. When work finishes, move it to Done.
Update `CHANGELOG.md` in the same change.

## Open

### Metrics audit follow-ups (2026-11-08)
- Held-out test bank: the 120-prompt bank leaves the reserve
  split empty (60/30/30/0). Grow the bank under a new
  `BANK_VERSION` so reserve prompts exist for re-verify and a
  read gate that never sees tuning or final prompts. Blocked on
  prompt authoring, not code.
- Persistent final-split ledger: the ledger lives in one
  process and the seed is caller-free, so seed shopping across
  runs is possible. Needs an on-disk ledger keyed by trait,
  model, bank hash, and a pinned seed set. Product decision on
  the seed policy first.
- Verdict gating on headline eligibility: fake/noop ports, fast
  schedule, and unenforced guard now flag a run as
  headline-ineligible in `metrics.comparability`, but the knob
  verdict itself is unchanged. Guard is off by default in every
  caller today, so gating the verdict on it would void all
  current knobs. Needs a product decision.
- Request-to-artifact latency: `report.seconds` covers compute
  stages only. Model load, GPU queue wait, and follow-up jobs
  need their own timers before any user-visible speed claim.
- Independent scoring: one judge family scores effect and
  fluency. A judge-swap or human check needs fresh generations
  and a pre-registered sampling protocol. The protocol is
  undecided.
- Guard coverage: Create guard checks only the +dose vector at
  24 tokens. update_down and delete are unchecked, and the
  guard prompt sets are small and fixed.
- Down control mismatch: update_down (negative vector addition)
  is carded against a shuffled ablation control, not a shuffled
  negative-dose vector. The control strength does not track
  dose. Fixing it changes down/delete numbers, so it needs its
  own change and re-measurement.
- Signed dose consumers: GGUF export and re-verify now use
  `patch.signed_dose`. Combine, Pareto, and repair follow-ups
  still read the tuned magnitude; audit each before a down
  knob ships through them.
- Repeated timing protocol: speed claims still rest on single
  runs. A fixed-environment repeated-timing protocol (median
  of N, same machine state) is needed before ranking small
  speed differences.
- Real-weights timing baseline: no wall-clock split (generation
  vs judge vs capture vs IPC) has been measured on weights.
  The counters now exist; one timed Create on the model lane
  is the next step before ranking hot paths.

### SH: ship to 10 (specs/SH-ship-to-10.md)
- Phase A done 2026-11-08: PR #61 merged to develop, gate on
  the merged tree passed (873 tests), develop merged to main
  (PR #63).
- Phase B done as a result, not as a pass: rude retrain at 90
  questions and the dose cap both fail the fluency tail at 50
  fresh questions. The rude mix ships as a no-knob and is
  guard-blocked (Research-only). Catholic passes at 50. Pirate
  fails at 50 (gain 0.225). Report: `roadmap/reports/SH.md`.
- Phase C done except live hosting: no-dial state for learned
  mixes, multi-patch registry and re-verify, Research owner
  switch with audit records, `/health` and the README server
  section. Dino 2026-11-08: local use for now; a hosted run
  waits until a host is wanted.
- Pirate taken down 2026-11-08 on Dino's order: it failed its
  50-question confirm (gain 0.225). `taken_down.json` in its
  job dir records why; the app offers no pirate knob.
- Known gap: the agent still sets `hand_validated` when it
  registers a family from its own labels. The committed store
  now records the label source and claims no hand validation
  for agent labels. Renaming the gate field waits for a
  contract change.

### Re-confirm the learned rude mix under the facet judge
- Done 2026-11-06. FAIL on `fresh_questions_v5.json` (20 new
  questions): gain 0.182 (bar 0.30), fluency p10 0.101 (bar 0.30).
  The mix is condescending, not hateful. No learned knob.
- Follow-up done 2026-11-06: few-shot examples fixed the strength
  (real rude text, facet gain to 0.478). No scale passes both bars
  on a fresh set: the fluency tail (1-3 of 20 prompts repeat) fails
  at every strong scale. Best rows: 0.625 gain 0.270 / p10 0.630;
  0.65 gain 0.257 / p10 0.288.
- Next levers tried in SH Phase B (2026-11-08): 90 training
  questions and a per-prompt dose cap. Both fail the fluency
  tail at 50 fresh questions. Closed: the family ships as a
  no-knob; the mix is guard-blocked, Research-only.

### A patch format for learned vector mixes (ML follow-up)
- Done 2026-11-06: `kind: "multi"` patches hold one raw vector per
  layer. The server offers a passed learned confirm as a knob
  (dose 1.0 = tested mix) in the workbench and chat. The rude mix
  stays guard-blocked in normal mode: the mean push also weakens
  refusals (min_ratio 0.0). Research mode runs it and records
  `would_block`. The read dial still needs one layer, so learned
  traits have no dial yet.

### Bayesian optimization for the mix search (GRACE)
- GRACE cuts the steering test count by about 40% with a Bayesian
  optimizer over layer and dose. Deferred until the greedy step log
  shows the call count hurts.

### A patch format for combined knobs
- `knob.lobot` holds one layer. A passing multi-layer mix needs a
  format that holds many layers, plus install and re-verify paths.
- Deferred until a combined search passes its fresh-prompt confirm.
  A failing mix needs no format.

### Agent llama lane crashes with a warm torch model (bug, FIXED 2026-11-05)
- Symptom: `AcceleratorError: index ... is out of bounds` in the
  agent run, after Muse drafts the behavior. Seen on the Catholic
  run, Qwen 4B. Same crash shape as the 2026-11-04 agent crash.
- Cause: the New behavior screen checks the llama.cpp lane box by
  default. The agent then starts a `lobot-llamad` copy of the
  model while the torch copy stays warm in the same server. Two
  4B stacks on a 24 GB Mac corrupt MPS memory; the failure is a
  garbage tensor index, not a clean out-of-memory error.
- Workaround until fixed: uncheck the llama lane box (torch lane,
  slower), after a clean server restart. A poisoned process does
  not heal; restart first.
- Fix options to pick from: refuse the llama lane with a clear
  error while a torch model is warm; unload the torch model
  between agent phases; or run the daemon on a 4-bit GGUF so both
  stacks fit. The GPU slot count must treat daemon plus torch in
  one process as two stacks.
- Follow-up (same day): one active stack at rest. Create jobs and
  agent runs now close their lane daemon when the job ends
  (`create_job` finally, agent route finally). Before, agent
  daemons stayed warm for the life of the server.
- Deferred: phase-split Create. Run torch alone for capture, close
  it, run the daemon alone for generation and scoring, reload
  torch for the guard. One stack at any moment, most of the llama
  speed kept, at the cost of two model reloads per Create. This
  is the clean fix for small-memory machines; it restructures how
  `trait_eval` stages call the lane, so it needs its own spec.
- **Fixed**: `llama_backend.check_stack` refuses the daemon start
  when GGUF bytes plus warm torch bytes pass 14 GB, with the way
  out in the message (torch lane, or `LOBOT_CREATE_QUANT=q4_k_m`).
  All three daemon call sites pass the torch size. Regression
  tests: `tests/test_llama_stack_guard.py` (5 tests; the Qwen
  sizes refuse, the Llama and q4 sizes pass, and the daemon
  process never starts when refused).

### Remove and rerun a behavior creation (planned 2026-11-05)
- Wanted from the UI: delete a created behavior, and run its
  creation again, without going back to the command line.
- Open choices to settle before building:
  - "Remove" deletes what, exactly: the job evidence, the family
    in `store/traits.json`, the knob patch file, or all three?
    The audit log is a record and must stay.
  - "Rerun" reuses what: the same family text and splits (a true
    re-check), or fresh splits (a new run that can differ)?
  - Where the buttons live: the evidence page, the shelf, or both?
- Guard rule stays: a rerun runs the full guard. Research mode is
  not a shortcut around a failed run.

### Fix the fluency tail of the v2 winner (ML follow-up)
- Done in v2 (2026-11-05): the probe score ranks steering, and
  layer 12 ranks 3 of 28. The harder push ran (0.01 threshold,
  doses to 1.5). Winner: L11 at 0.5 alone.
- On 20 brand-new questions its gain is 0.572 (bar 0.30, pass),
  but worst-decile fluency is 0.026 (gate 0.3, fail). The effect
  is real now. The tail of the text breaks.
- Next idea, untested: search with the confirm-style tail in the
  loop (rank mixes by worst-decile fluency on more search
  questions), or cap the dose per answer length. Any new winner
  needs a new fresh set; `fresh_questions_v2.json` is spent now.

### Finer dose grid for weak but clean knobs (pirate)
- Pirate at dose 1.0 is strong but breaks some answers; at 0.5 it
  is clean but weak. Doses 0.25 steps apart may hold a middle dose.
- Deferred: the ML dose tune tries 0.25 steps on mixes first. Its
  result shows if a finer single-layer grid is worth a Create run.

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

### Live check of judged install verify on Ollama (IN)
- llama.cpp ran live on 2026-11-02 (pirate v4, Q4_K_M: measured +0.40 on
  16 prompts, shipped [+0.19, +0.48], HOLDS; reports/UX2.md). Ollama is
  not installed on the dev Mac, so its path never ran live.

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

- **Learned steering vectors** (ML step 3, done 2026-11-06):
  `learned.py` trains one push per layer with a drift cost. First
  fresh-confirm pass for the evil family (gain 0.402, worst-decile
  fluency 0.392 at scale 0.625). Report: `roadmap/reports/ML.md`.
- **Open-ended Create** (deferred 2026-10-29): built as C3 open
  descriptions; noir and cowboy passed; the Muse Spark agent automates it.
