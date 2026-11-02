# lobot roadmap: from a workbench to behavior patches you can trust

Goal (1000X): anyone can see a behavior inside an open model, turn it into a
small patch, prove what the patch does, check that it is safe, and share it so
others can install it and re-check the proof on their own machine.

This folder holds the plan and one spec per workstream. Each spec is written so
one agent can do the work alone, check it, and know when it is done.

- `contracts.md`: the shared interfaces (API routes, Python signatures, file
  formats). Specs build against these. Change a contract only through its owner.
- `specs/`: one file per workstream.
- `later.md`: the 100X and 1000X steps after the demo.

## Stages

| Stage | When | Outcome |
|---|---|---|
| **10X** | now to Mon 2026-11-03 (internal demo deadline; Tue 11-04 is buffer) | The loop on Llama, on screen: read a neuron, steer it, guard it, with proof on screen and a patch you can export. 3-minute video. |
| **100X** | November to January | Describe a behavior in words and get a tested patch in about a minute; clean features (not single neurons); live dials in chat; install into llama.cpp and Ollama; patches anyone can re-verify. |
| **1000X** | 2027 | An open, signed registry of behavior patches for open models, with portable evidence, guards for many risks, cross-model transfer, and behavior CI. |

## Decisions (2026-10-30)

| Topic | Decision |
|---|---|
| Hero model | Llama 3.2 3B Instruct. Qwen 3.5 4B stays supported and is the code default until U1 makes the model choice persistent. |
| Demo format | A recorded video. Replay mode plays every on-screen run from real recordings, so the take cannot fail. A live take is a bonus, not a must. |
| Judge | Local judge by default. A judge from the Meta model API may score offline evidence before the demo (J1), never live. |
| Deadline | External deadline: Tue 11-04. Target: final video ready **Mon 11-03 23:00 ET**. Tuesday is buffer only. |

## The demo we are building (10X)

Three beats, on Llama 3.2 3B, all local:

1. **Read.** Click a word. lobot shows the neurons behind it, what they fire on,
   and which words they push toward.
2. **Steer.** Push one of those neurons. The answer forks live. On screen:
   "the reader predicted *X*; steering made *X* N× more likely".
3. **Guard.** Push too far. The refusal guard shows BLOCKED with the measured
   reason. Lower the dose; it passes. Export the patch with its evidence.

Pitch: "See it, change it, prove it, keep it safe, on your own computer."
Do not pitch "we name neurons"; Transluce and Neuronpedia already do.

## Workstreams and parallel lanes

The Mac holds **one model in memory at a time**. So work is split into three
kinds of lane:

- **No-model lane**: code and tests run on fake or tiny models, or on replay.
  Any number of agents can work here at once.
- **Model lane**: needs real weights. Only **one** agent at a time. Book it in
  `docs/roadmap/model-lane.md` before loading weights (see below).
- **UI lane**: Claude (UI lead) owns every screen.

Status as of 2026-10-30 (night 1 and day 2). Reports: `reports/<ID>.md`.

| ID | Workstream | Owner | Lane | Spec | Status |
|---|---|---|---|---|---|
| R1 | Reader core library | Backend agent | No-model | `specs/R1-reader-core.md` | **Done** |
| R2 | Reader API (reply words in /api/neurons, neuron card, find, lift) | Backend agent | No-model | `specs/R2-reader-api.md` | **Done** (+ `reply_char`, Claude) |
| R3 | Reader validation and demo-neuron search | Model agent | Model | `specs/R3-reader-validation.md` | **Done**: demo neuron L19N4330 (storm), 10 recorded cards |
| D1 | Read dials: baseline + per-token lists | Backend agent | No-model | `specs/D1-read-dials.md` | **Done** (model check: means 0.53 vs 16.56) |
| E1 | Evidence and patch API | Backend agent | No-model | `specs/E1-evidence-api.md` | **Done** |
| S1 | Guard hardening and demo guard | Backend agent, then model slot | Both | `specs/S1-guard-hardening.md` | **Done**: 0 misses; storm blocks are real refusal loss (`reports/S1-followup.md`) |
| P1 | Patch export and install | Backend agent, then model slot | Both | `specs/P1-patch-install.md` | **Partly done**: export + verify pass; install stopped (no llama.cpp/Ollama on the Mac) |
| C2 | Hostility family (validated judge 1.000) | Backend agent | Model | `reports/day-2-backend.md` | **Done**; its Create was stopped, not run |
| J1 | Meta-model judge, offline | Backend agent | No-model (network) | `specs/J1-meta-judge.md` | **Done**: no logprobs, fallback (a); meta matches local 1.000 / 0.917 (`reports/J1.md`) |
| U1 | UI: Create, Update, guard, evidence | Claude (UI lead) | UI | `specs/U1-ui-wave1.md` | **Done** |
| U2 | UI: reader in the inspector, find, lift proof | Claude (UI lead) | UI | `specs/U2-ui-reader.md` | **Done** |
| U3 | UI: read dials on the tree | Claude (UI lead) | UI | `specs/U3-ui-dials.md` | **Done** |
| V1 | Recordings and demo production | Human + model slot + Claude | Both | `specs/V1-demo.md`, `demo-script.md` | **In progress**: recordings and script done; dry run on the Mac, then record |

Status as of 2026-10-31 (100X). Reports: `reports/<ID>.md`.
Note: 100X `C2` (fast Create) reuses the 10X `C2` label (hostility
family, reported in `reports/day-2-backend.md`); the rows below are
the 100X workstreams from `later.md`.

| ID | Workstream | Owner | Lane | Spec | Status |
|---|---|---|---|---|---|
| F1 | Clean features | Backend agent | Model | `specs/F1-clean-features.md` | **Done**: L12+L20 SAEs, OVERALL PASS (coh 0.833/0.909, lift 49-50/50) |
| C2 | Fast describe-to-patch | Backend agent | Model | `specs/C2-fast-create.md` | **Done, acceptance FAILED**: median 158.5 s (> 120 s), parity 0/3. Llama Create lane (2026-10-31): full schedule 2043 s → 480 s (4.3x) with identical verdict, layer, dose, eligibility; still above the 2-minute gate |
| C3 | Open descriptions | Backend agent | No-model (network) | `specs/C3-open-descriptions.md` | **Done, PASS 2/2 after fix**: judge-phrase fix; noir 0.85 + knob, cowboy 1.00 + knob (read gate). Rhyming stays out: the 3B judge cannot see rhyme (0.75 even with the ideal label) |
| CH | Chat with live dials | Backend agent | No-model | `specs/CH-chat-dials.md` | **Done**: replay 10/10 exact, dial diff 0.0 |
| DL | Delete | Backend agent | Both | `specs/DL-delete.md` | **Done**: pirate card PASS (mean 0.360), NLL +0.035, guard verified |
| IN | Install everywhere | Backend agent | No-model | `specs/IN-install.md` | **Done**: llama.cpp differs + 8/8 recorded tokens. Judged verify now uses the patch's final split as a mean (`judged_n`); not run live (no llama-cli/ollama on the Mac) |
| RV | Re-verify anywhere | Backend agent | Both | `specs/RV-reverify.md` | **Done**: pirate quick PASS 0.745 in [0.728,0.812], route/upload/script agree; open-family patches verify (judge label from the patch) |
| GX | Guards for more risks | Backend agent | Model | `specs/GX-guards.md` | **Done**: 0 misses all screens held-out; honesty+privacy caught flips; harmful vacuous/opt-in |
| CB | Combined patches | Backend agent | Model | `specs/CB-combined.md` | **Done**: shakespeare+noir pass alone, union flips honesty (0.442); FP findings reported |

Status as of 2026-10-31 (1000X). Report: `reports/1000X.md`.

| ID | Workstream | Status |
|---|---|---|
| RG | Signed patch registry | **Done**: ed25519 entries, tamper/trust/dual-use refusals; publishes only full-generation guard verdicts; cowboy published live |
| AU | Audit trail | **Done**: hash-chained log, chain check, history rebuild |
| CI | Behavior CI | **Done, live**: cowboy evidence holds on torch (0.267) and llama Q4_K_M (0.249) in [0.137, 0.391]; no live breaking case yet |
| XM | Cross-model transfer | **Done, live**: pirate Llama (L17 @1.5) → Qwen (L21 @1.0, knob, read 1.00) on one benchmark |
| SC | Scale (70B) | **Not started**: does not fit 24 GB |

Status as of 2026-11-01 (real 100X, first three pieces).

| ID | Workstream | Spec | Status |
|---|---|---|---|
| PP | Prompt Pareto | `specs/PP-prompt-pareto.md` | **Done, live**: pirate and cowboy both `tradeoff`; a prompt beats the vector at every useful dose; pirate's shipped dose 1.5 has fluency 0.017 (`reports/PP.md`) |
| FD | Fluency-aware dose | `specs/FD-fluency-dose.md` | **Done, live, gate PASS**: pirate moved from a broken dose (fluency 0.017, card FAIL) to a fluent one (0.948, card PASS, gain +0.333); cowboy unchanged (`reports/FD.md`) |
| GR | Guard repair | `specs/GR-guard-repair.md` | **Built, gate FAILED**: gradient repair cut pirate refusal flips 7 → 1 with the trait kept, but no patch reached a clean guard (`reports/GR.md`) |

## Next version: patches that beat their prompt (v-next)

Owner of every workstream below: Muse (coding agent). Each spec is
written so one agent can build it alone, check it, and know when it is
done. Read [AGENTS.md](../../AGENTS.md) first.

### Thesis

Published work agrees on one hard fact: activation steering usually
loses to a plain prompt. AxBench (Stanford) found that a prompt beat
every steering method. SteerScope (2026) tested 23 methods; none got a
stronger effect without more side effects than a prompt. PP measured
the same on our patches (both `tradeoff`).

The goal of v-next: **every patch proves where it stands against its
own prompt, predicts its side effects, and repairs them.**

### Rules for all v-next work

1. **Prompt bar.** Every new patch method reports its PP row next to
   `prompt:label`. A method that loses says so, with numbers.
2. **Gates first.** Each spec fixes its acceptance gate before any
   live run. Do not change a gate after you see results. If a gate
   was wrong, write a new version of the spec and say why.
3. **Held-out data stays held out.** Tuning and fit data never
   confirm a claim. The guard test set (GH) gets one look per method
   and patch, logged.
4. **One model job at a time** (AGENTS.md). Log every run in
   `model-lane.md`.
5. **Honest reports.** Separate "tested with fakes" from "tested on
   real weights". Report failed gates with numbers.

### Workstreams, in order

| # | ID | Workstream | Spec | Depends on | Lane | Days | Basis |
|---|---|---|---|---|---|---|---|
| 1 | U4 | UI: evidence page v2 (prompt chart, fluency, all screens, repair button) | `specs/U4-ui-evidence-v2.md` | PP, FD, GR | UI + no-model | 2 | **Done** (`reports/U4.md`); checked in replay mode, follow-up buttons tested with fakes |
| 1b | UX | Frontend 100x: dose dial with recorded replies, tuning grid, Library, Muse explanation | `reports/UX-100x.md` | U4 | UI | 1 | **Done**; checked in replay mode |
| 1c | UX2 | 100X features on screen: describe in words (Muse agent, self-labeling), chat with dials, re-check, removal, install; key legend; stream-disconnect lock fix | `reports/UX2.md` | UX | UI + model | 1 | **Done**; see the report for what ran on real weights |
| 1d | UX3 | Last screens on real weights (removal view, Muse suggestions, judge check); removal `complete` fix; app guide and agent docs | `reports/UX3.md` | UX2 | UI + model | 1 | **Done**; every screen ran live except Ollama |
| 0 | MOD | Modularity: one place to change each part (guard, judge, direction, dose rule, Create stages, storage, settings, server routers, frontend modules) | `specs/MOD-modularity.md` | none | No-model + one parity run | 4-6 | (engineering) |
| 2 | JS | Judged scores: dose rule without NLL; unrequested-change search | `specs/JS-judged-scores.md` | FD, PP | Both | 3 | AxBench (Stanford), VibeCheck (Berkeley) |
| 3 | GH | Fresh guard test set; repair v3 (iterated gradient projection) | `specs/GH-guard-holdout-repair-v3.md` | GR, FD | Both | 2 | "Safety cost of steering vectors is separable and reducible" (2026) |
| 4 | UD | "What it thinks of you" dashboard in chat | `specs/UD-user-model-dashboard.md` | CH | Both + UI | 3 | TalkTuner (Harvard) |
| 5 | HD | One-second draft: description to vector | `specs/HD-one-second-draft.md` | C3, JS, PP | Both | 4 | HyperSteer (Stanford), REMEDI (MIT) |
| 6 | TD | Trained directions (two-way preference) | `specs/TD-trained-directions.md` | JS, PP | Both | 3 | RePS (Stanford), "Steer like the LLM" (2026) |
| 7 | SG | Scientist agent: predictions first | `specs/SG-scientist-agent.md` | agent, audit | No-model + network | 2 | MAIA (MIT), Predictive Concept Decoders (Transluce) |
| 8 | SA | Side-effect atlas and forecast | `specs/SA-side-effect-atlas.md` | PP, 6+ knobs | Both | 2 | "Forecasting side effects of activation steering" (2026) |
| 9 | TM | Truth meter; head-level honesty steering | `specs/TM-truth-meter.md` | GX, CH, GH | Both + UI | 3 | Geometry of Truth (MIT), ITI (Harvard) |
| 10 | LQ | Ask the activations (stretch, high risk) | `specs/LQ-latent-qa.md` | store | Model | 4 | LatentQA (Berkeley) |

Why this order:
- MOD first: it is a pure refactor, and it turns most later workstreams
  into "add one module and one setting".
- U4 first: the backend already measures more than the screen shows.
- JS before HD and TD: both new vector sources need a fair dose rule.
- GH before any new repair or honesty claim: the old guard prompts
  are spent as a test set.
- UD is the strongest demo and needs no new vector method.
- SA needs at least 6 passing knobs; JS and TD should add some.

No-model parts can run in parallel (code and tests with fakes). Model
runs go one at a time in the order above.

### Sources

- Stanford: [AxBench and steering](https://cs.stanford.edu/~wuzhengx/steer/index.html), [HyperSteer](https://arxiv.org/html/2506.03292v1), [RePS](https://arxiv.org/abs/2505.20809), [ReFT](https://cs.stanford.edu/~wuzhengx/reft/index.html)
- Berkeley: [LatentQA](https://iclr.cc/virtual/2026/poster/10007461), [VibeCheck](https://sky.cs.berkeley.edu/project/vibecheck/)
- MIT: [MAIA](https://multimodal-interpretability.csail.mit.edu/maia), [REMEDI](https://arxiv.org/pdf/2304.00740), [Geometry of Truth](https://arxiv.org/pdf/2310.06824)
- Harvard: [TalkTuner](https://arxiv.org/pdf/2406.07882), [ITI](https://arxiv.org/abs/2306.03341)
- Transluce: [Predictive Concept Decoders](https://arxiv.org/abs/2512.15712)
- 2026: [Forecasting side effects](https://arxiv.org/abs/2608.11227), [Safety cost separable](https://arxiv.org/pdf/2608.08383), [SteerScope](https://arxiv.org/abs/2610.07722), [Steer like the LLM](https://arxiv.org/pdf/2605.03907)

No Caltech paper on steering or interpretability was found for this
plan. Only abstracts were read; no paper was reproduced.

## Next steps

Open work outside v-next, each with its reason, lives in
[../TODOS.md](../TODOS.md): the C2 two-minute gate (HD attacks it), a
live judged install check on Ollama (IN), a live "model update breaks a patch"
case for behavior CI, a fairer privacy rule, the O6 chat-cache
decision, and 70B scale (SC, needs rented GPUs).

The 10X schedule (Thu 10-30 to Tue 11-04) is in this file's git history.

## Rules for agents

Branches, merges, tests, contracts, honesty, style, commits, the model
lane, reports and the definition of done: [AGENTS.md](../../AGENTS.md).
