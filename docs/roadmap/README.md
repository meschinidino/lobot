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
| C2 | Fast describe-to-patch | Backend agent | Model | `specs/C2-fast-create.md` | **Done, acceptance FAILED**: median 158.5 s (> 120 s), parity 0/3; full verified unchanged; default stays full |
| C3 | Open descriptions | Backend agent | No-model (network) | `specs/C3-open-descriptions.md` | **Done, acceptance FAILED 1/2**: rhyming 0.55, noir 0.85 + knob; flow + noir family merged |
| CH | Chat with live dials | Backend agent | No-model | `specs/CH-chat-dials.md` | **Done**: replay 10/10 exact, dial diff 0.0 |
| DL | Delete | Backend agent | Both | `specs/DL-delete.md` | **Done**: pirate card PASS (mean 0.360), NLL +0.035, guard verified |
| IN | Install everywhere | Backend agent | No-model | `specs/IN-install.md` | **Done**: llama.cpp differs + 8/8 recorded tokens; judged interval FAIL (see report) |
| RV | Re-verify anywhere | Backend agent | Both | `specs/RV-reverify.md` | Pending |
| GX | Guards for more risks | Backend agent | Model | `specs/GX-guards.md` | Pending |
| CB | Combined patches | Backend agent | Model | `specs/CB-combined.md` | Pending |

## Next step: J1 step 0, can the Meta judge say "how sure"?

The local judge scores an answer as P(Yes) / (P(Yes) + P(No)), read from the
model's own next-token probabilities. The report cards need that smooth score.
Before building J1, check what the Meta model API returns:

1. With `META_API_KEY` set, send one judge question and look for token
   log-probabilities in the response (often called `logprobs` / `top_logprobs`).
2. If it returns them: build J1 as specified, computing P(Yes) the same way as
   the local judge. About 1 hour.
3. If it returns only text: pick one fallback and say which in the report:
   (a) sample the same question N times (for example 10) at temperature 1 and
   use the share of Yes answers; (b) ask for a 0-100 number (weaker: a stated
   guess, not a probability). Re-run the hand-label agreement check either way.
4. Report in `reports/J1.md`: the API response shape, the method chosen, and
   agreement with the hand labels next to the local judge's.

Contracts let R2, D1, E1, U1, U2 and U3 start **today**, in parallel, before
R1 or any model run is finished. Backends implement the contract; the UI builds
against it with replay and stubs.

## Order and schedule (10X)

```
Thu 10-30  R1, R2, D1, E1, S1(code), P1(code), J1 | U1
Fri 10-31  R3 (model slot) | merge R1/R2/D1/E1 | U2, U3 | S1 measure (model slot)
Sat 11-01  R3 shortlist done | P1 round trip (model slot) | U2/U3 against real API
           feature freeze 23:00 ET
Sun 11-02  V1 recordings (model slot) | full test pass | bug fixes only
Mon 11-03  V1 video cut, share card, post text | final video by 23:00 ET
Tue 11-04  buffer: re-cut only if a check fails
```

Model-lane order (one at a time): R3 search → S1 measurement → P1 round trip →
V1 recordings. Muse's real-model checks fit between them.

## Rules every agent follows

1. **Branches.** Three kinds only: `main` (stable, protected, changes only by
   pull request from `develop`), `develop` (integration), and feature branches
   `feature/<ID>-<short-name>` (for example `feature/J1-meta-judge`) cut from
   `develop`. Before merging: `git fetch origin && git merge origin/develop`,
   run the full tests, push the feature branch. Never rebase a shared branch.
   Never force-push.
2. **Merge.** A feature branch merges into `develop` with a merge commit once
   the full tests pass; then delete the feature branch. `develop` goes to `main`
   by pull request when a milestone is stable (for example the recorded demo).
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
