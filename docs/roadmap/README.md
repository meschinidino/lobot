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
| **10X** | now to Wed 2026-11-04 (demo deadline) | The loop on Llama, live: read a neuron, steer it, guard it, with proof on screen and a patch you can export. 3-minute video. |
| **100X** | November to January | Describe a behavior in words and get a tested patch in about a minute; clean features (not single neurons); live dials in chat; install into llama.cpp and Ollama; patches anyone can re-verify. |
| **1000X** | 2027 | An open, signed registry of behavior patches for open models, with portable evidence, guards for many risks, cross-model transfer, and behavior CI. |

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

| ID | Workstream | Owner | Lane | Depends on | Spec |
|---|---|---|---|---|---|
| R1 | Reader core library | Backend agent | No-model | none | `specs/R1-reader-core.md` |
| R2 | Reader API (neuron card, find, word neurons) | Backend agent | No-model | contracts only | `specs/R2-reader-api.md` |
| R3 | Reader validation and demo-neuron search | Model agent | Model | R1 | `specs/R3-reader-validation.md` |
| D1 | Read dials in /api/run | Backend agent | No-model | contracts only | `specs/D1-read-dials.md` |
| E1 | Evidence and patch API | Backend agent | No-model | contracts only | `specs/E1-evidence-api.md` |
| S1 | Guard hardening and demo guard | Backend agent, then model slot | No-model, then Model | none | `specs/S1-guard-hardening.md` |
| P1 | Patch export and install | Backend agent, then model slot | No-model, then Model | E1 contract | `specs/P1-patch-install.md` |
| J1 | Meta-model judge, offline | Backend agent | No-model (network) | model decision | `specs/J1-meta-judge.md` |
| U1 | UI: wave-1 wiring (Create, Update, guard, evidence) | Claude (UI lead) | UI | contracts | `specs/U1-ui-wave1.md` |
| U2 | UI: reader in the word inspector, find-a-neuron | Claude (UI lead) | UI | contracts | `specs/U2-ui-reader.md` |
| U3 | UI: read dials on the tree | Claude (UI lead) | UI | contracts | `specs/U3-ui-dials.md` |
| V1 | Recordings and demo production | Human + model slot + Claude | Model + UI | R3, S1, U1-U3 | `specs/V1-demo.md` |

Contracts let R2, D1, E1, U1, U2 and U3 start **today**, in parallel, before
R1 or any model run is finished. Backends implement the contract; the UI builds
against it with replay and stubs.

## Order and schedule (10X)

```
Thu 10-30  R1, R2, D1, E1, S1(code), P1(code), J1 | U1
Fri 10-31  R3 (model slot) | merge R1/R2/D1/E1 | U2, U3 | S1 measure (model slot)
Sat 11-01  R3 shortlist done | P1 round trip (model slot) | U2/U3 against real API
Sun 11-02  feature freeze 18:00 ET | V1 recordings (model slot) | full test pass
Mon 11-03  V1 video cut, share card, post text | bug fixes only
Tue 11-04  rehearse twice, final video by 18:00 ET (3 h buffer)
```

Model-lane order (one at a time): R3 search → S1 measurement → P1 round trip →
V1 recordings. Muse's real-model checks fit between them.

## Rules every agent follows

1. **Branch.** Start from `claude/admiring-mendel-mn88n8` (the integration
   branch). Work on `lane/<ID>-<short-name>`, for example `lane/R1-reader-core`.
   Before you ask for a merge: `git fetch origin && git merge
   origin/claude/admiring-mendel-mn88n8`, run the full tests, push your lane
   branch. Never rebase a shared branch. Never force-push.
2. **Merge.** Claude (integrator) merges lane branches into the integration
   branch after review. A lane is not merged with failing tests.
3. **Tests.** `uv run python -m unittest discover -s tests -t .` must pass.
   Every spec names its own new tests; write them first or with the code.
   Model-backed tests go in the model-lane test group and skip without weights.
4. **Contracts.** Build against `contracts.md` exactly. If you need a change,
   write it in your branch's `contracts.md` diff and say so in your report; do
   not change another lane's behavior silently.
5. **Honesty.** Never claim "verified" for simulated or illustrative results.
   Label illustrative values on screen and in reports. A gate that fails is
   reported as failed, with numbers.
6. **Style.** Simplified Technical English in docs, reports and UI text: short
   sentences, active voice. No event or competition references in repo files.
7. **Commits.** Author is the repo's configured user. No co-author or
   tool-attribution lines.
8. **Model lane.** Before loading weights, add a line to
   `docs/roadmap/model-lane.md`: `<date time> <agent> <ID> <model> start`.
   When done: `... end`. If another agent holds the lane, wait or work on the
   no-model part.
9. **Report.** When done, write `docs/roadmap/reports/<ID>.md`: what changed,
   test counts, acceptance results (exact command output), known gaps.

## Definition of done (all specs)

- Every acceptance check in the spec passes, with output pasted in the report.
- Full test suite passes; new tests exist for every new behavior.
- `docs/api.md` updated for any new or changed route.
- No new lint or type errors in touched files; no dead code left behind.
- Report written; lane branch pushed; integrator notified.
