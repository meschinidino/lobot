# lobot 10X plan: behavior knobs you can see, measure and export

> **Status (2026-10-30):** the goal and schedule now live in `docs/roadmap/`. This plan stays the record of reviews and binding decisions (D1-D14, design decisions 1-14). Where they differ, the roadmap wins on scope and order; this plan wins on engineering decisions.

Working plan for /plan-ceo-review, 2026-10-29. Branch: main (no remote). Review depth: strategy-only (the user asked for scope and direction, not implementation design). Mode: SCOPE EXPANSION (explicit user request).

Source of truth for problem and constraints: office-hours design doc `~/.gstack/projects/model-lobotomy/dinomeschini-unknown-design-20261028-233909.md`.

## Concept under review (as requested)

1. Describe a behavior in plain words.
2. lobot builds local contrast prompts (same question bank, two instructions).
3. lobot finds a residual-stream steering vector (single-neuron mode stays).
4. Live, batched multi-dose filmstrip: one prompt at many doses at once.
5. Click a word to see which neurons fired.
6. Measured proof: self-judge with a random-vector control, fluency, factual drift.
7. Replayable recipe.
8. Export as a llama.cpp control vector (1000X: training-free behavior editing, tiny shareable patches).

Hero model: Llama 3.2 3B. Constraints: local only, Apple M4 Pro 24 GB. Goals: a strong first 10 seconds in a short demo video, shareable on X, serious and useful.

## Pre-review system audit

- Recent work: `7d0fe0e` web UI + DESIGN.md, `ad5d42a` Stop. Uncommitted: `scripts/spike_knob.py`.
- No TODOS.md, no stash, no remote, no binding decisions in `gstack-decision-search`.
- Prior learnings applied: STE writing style (user-stated, 10/10); BPE split offset includes the leading space (observed, 9/10).
- Retrospective: v1 office-hours deferred filmstrip, recipe replay and neuron graph to v1.5. This plan picks up filmstrip and replay; the neuron graph becomes "click a word".
- DESIGN_SCOPE: yes (new screens: describe, filmstrip, proof panel, export).
- Taste references: good: `server.py` stream contract + lock handling, `tests/test_server.py` (stubbed engine, explicit no_claim), `scripts/spike_knob.py` (pre-registered verdict). Avoid: `app.py` mixing Gradio UI with engine state (server.py imports it as the engine), the decorative jitter in the agreement trace (not data).

## Landscape (Layer 1-3)

- **Layer 1 (tried and true):** contrastive activation addition / representation engineering: mean-difference vectors from contrast prompts, added to the residual stream. repeng trains control vectors in under a minute and exports GGUF for llama.cpp. [sources: sourcepulse repeng page; jisaku glossary, secondary]
- **Layer 2 (current):** EasySteer (web demo: inference, chat, vector extraction, side-by-side baseline vs steered), Saklas (localhost dashboard, concept -> contrast -> direction), Dialz (toolkit). SteerScope 2026: steering does not yet beat prompt steering on efficacy vs side effects; side effects grow out of distribution. Steering Vector Fields (arXiv 2602.01654): some concepts barely respond, steering can backfire on some inputs.
- **Layer 3 (first principles):** the field compares steering to prompting and steering often loses. So the honest product is not "steering instead of prompting". It is a tool that shows, per behavior, when a knob beats a prompt and when it does not, and what a knob gives that a prompt cannot: a continuous dose, composition of several knobs, a change that a later user message cannot talk the model out of, and an inspectable mechanism.

EUREKA (pending user decision D1): every steering tool hides the comparison with prompting; lobot can make it the headline.

## 0A. Premise challenge

- **Real problem (design doc):** the user wants to play with model internals in a friendly, visual way; the user's 10X bar was "graphically astounding, lightning fast, crystal clear" and "for everyone to be able to judge".
- **Target outcome:** a viewer understands in 10 seconds that a measured internal knob changed a model's behavior, and a skeptic can verify it.
- **Proxy risk:** "describe -> knob" alone is a proxy. Existing tools already do it. Without a comparison with prompting, a serious viewer reads it as "a slower system prompt".
- **Do-nothing cost:** lobot stays a single-neuron replay tool for recipes found elsewhere; the whoa depends on one known preset (evil-qwen).

## 0B. Existing code leverage

| Sub-problem | Reuse | Notes |
|---|---|---|
| Contrast capture, direction | `scripts/spike_knob.py` capture_last, separability; `scripts/derive_llama_recipe.py` capture_means | Promote to a module; spike code is proven on Llama 3B |
| Residual vector hook | `spike_knob.add_vector` | New engine mode next to `steer.register_steering_hooks` (`steer.py:254`, mode "add") |
| Dose scale | `steer.calibrate_layer_medians`, spike `median_norms` | Same rule: dose x median residual norm |
| Batched multi-dose | none | New: per-row dose in the hook, batch generate |
| Streaming + Stop | `server.py` NDJSON stream, `steer.CancelCriteria` (`steer.py:391`) | Extend stages; Stop already per token |
| Judge + control | `spike_knob.judge`, random control | Promote; drop NLL and A/B probe as gates (spike v2) |
| Recipe string | `steer.recipe_string` | Needs a vector field and a parser (replay) |
| Export | none; llama.cpp not installed here | New; verify format against llama.cpp source |
| UI | `web/index.html`, DESIGN.md | New panels; capture view reused for X |

Rebuild avoided: engine stays in `steer.py`; no new framework.

## 0C. Dream state

```
  CURRENT STATE                    THIS PLAN                          12-MONTH IDEAL
  replay known neuron recipes  ->  describe -> knob -> measured   ->  open patch format: anyone
  one dose per run, 13 s,          filmstrip vs prompt, export         edits a local model's
  no discovery, no judge           to llama.cpp                        behavior in seconds, with a
                                                                       report card, and shares it
```

## Reality rows

| Row | Result | Receipt |
|---|---|---|
| premise-table | finding: unverified=3 (incl. 5/6 knob claim, corrected after spec review) (llama.cpp control-vector format and layer indexing; batched hooks fit in 24 GB) | see Premise table |
| already-done | pass: no commits for vectors, filmstrip or export; spike script uncommitted | `git log --grep`, no hits |
| surface-check | finding: `llama-cli` / `llama-server` not installed; control-vector flags unverified | `command -v llama-cli` |
| binding-decisions | pass: none recorded | `gstack-decision-search --scope branch` empty |
| numbers | pass with tags | see Numbers |
| deferred-asks | pass: no linked issue | n/a |
| pr-count | pass: default one PR per wave (no `.gstack/ship-policy.json`) | |

### Premise table

| Premise | Claim | Receipt | Verdict |
|---|---|---|---|
| Neuron limit | up to 4 rows, dose -3..+3 | `app.py:38-39` | verified |
| Llama dims | 28 layers, 3072 hidden, 8192 neurons | `steer.py:103` | verified |
| Knobs exist for persona/style | Pre-registered v2 verdicts: paranoid CLEAN, blunt WEAK, 4 NONE; judge gains at coherent doses 0.05-0.44; random control < 0.08 (directions are specific). Larger gains (to 0.97) only at doses the coherence rule flags | `logs/spike_knob/v2/report.md:5-110` | partly verified; "5/6 behaviors work" is UNVERIFIED (author's reading of samples) |
| Stop per token | CancelCriteria | `steer.py:391` | verified |
| llama.cpp export | control vectors in llama.cpp | secondary sources only | unverified |
| Batched multi-dose on MPS | fits 24 GB | none | unverified |

### Numbers

- Spike v2 per-behavior time: 82-96 s [measured: logs/spike_knob/v2/report.md]
- Full run Qwen 2x150 tokens: 13.5-15.7 s [measured: live run 2026-10-29]
- Stop latency: 150-160 ms [measured: live run 2026-10-29]
- Batched 7-dose filmstrip in about one run's time [estimated; re-measure before gate]

## Decision ledger

| ID and owner | Contract and evidence | Current | Proposed | Status | Exact approval and scope |
|---|---|---|---|---|---|
| D1 approach (user) | Landscape + user questions (see history) | **D) Full CRUD**: create named traits, read live dials, update by steering, delete by directional ablation; knob-vs-prompt inside the proof | — | approved | User answer 2026-10-29: "D) Full CRUD (recommended)". Scope: product headline and capability set; implementation details stay pending |
| D2 safety stance (user) | Delete = directional ablation; refusal preset exists | **A) No limits + refusal-guard toggle, guard ON by default**; research-mode switch in UI turns it off; patches record guard state; refusal preset available in research mode only | — | approved | User answer 2026-10-29: "A) Guard on by default (recommended)", after asking for "no limits with a safety toggle". Scope: default and switch behavior |
| S3-1 patch format (Section 3) | T1: shared patches + pickle = code execution | **A) safetensors + JSON in a zip; never unpickle** | — | approved | User answer 2026-10-29: "A) safetensors + JSON (recommended)". Scope: `.lobot` storage and its rejection test |
| O1 outside voice | Codex finding, citations verified | see Outside voice table | — | approved | User answer 2026-10-29 (D4): "A) Apply (recommended)" |
| O2 outside voice | Codex finding, citations verified | see Outside voice table | — | approved | User answer 2026-10-29 (D5): "A) Apply (recommended)" |
| O3 outside voice | Codex finding, citations verified | see Outside voice table | — | approved | User answer 2026-10-29 (D6): "A) Apply (recommended)" |
| O4 outside voice | Codex finding, citations verified | see Outside voice table | — | approved | User answer 2026-10-29 (D7): "A) Apply (recommended)" |
| O5 outside voice | Codex finding, citations verified | see Outside voice table | — | approved | User answer 2026-10-29 (D8): "A) Apply (recommended)" |
| O7 outside voice | Codex finding, citations verified | see Outside voice table | — | approved | User answer 2026-10-29 (D9): "A) Apply (recommended)" |
| O8 outside voice | Codex finding, citations verified | see Outside voice table | — | approved | User answer 2026-10-29 (D10): "A) Apply (recommended)" |
| O6 outside voice | Chat cache semantics undefined | pending implementation choice | future-tokens-only (default) vs recompute | pending (non-blocking) | Owner: wave-2 lead; decide before wave-0 performance gate |

## Answered: D1 (approved: D, full CRUD)
Commitment comparison:

| Commitment | Source/approval or pending | Current | A | B | C | D |
|---|---|---|---|---|---|---|
| Create: named trait from a description, saved with report card | pending | no | partial | partial | partial | yes, library |
| Read: live per-token trait dial | pending | no | no | no | yes, headline | yes, headline |
| Update: steer with the direction | requested concept | in concept | yes, headline | yes, headline | yes | yes |
| Delete: directional ablation at run time | pending | no | no | no | no | yes |
| Prompt-only baseline + verdict | pending | no | no | yes | yes, in proof | yes, in proof |
| Live batched dose filmstrip | requested concept | in concept | yes | yes | yes | yes |
| Proof (judge + random control, side effects) | requested concept | in concept | yes | yes | yes | yes, also for delete |
| Export | requested concept | control vector | control vector | control vector | control vector | control vector (update); baked weights for delete pending D-scope |
| Safety exposure (refusal deletion) | D2 pending | low | low | low | low | high: needs D2 |

Question: D1 — D1: How far should lobot go: edit only, edit vs prompt, read and edit, or full create-read-update-delete for model behaviors?
Header: D1 Approach
A) Edit-first as requested
Describe a behavior, get a knob, filmstrip, proof, export. Effort L, risk medium. ✅ The plan you described, with the fewest new parts ✅ A proven demo: paranoid and Shakespeare knobs already work. ❌ Overlaps existing tools, and experts can say a system prompt does the same.
B) Edit + knob vs prompt
A plus a prompt-only column and a verdict per behavior. Effort L, risk medium. ✅ Answers the skeptic honestly on screen ✅ Shows the knob's real edges: dose, stacking, outside the prompt. ❌ Still an editing tool at heart: an edge, not a new paradigm.
C) Read and edit (RU)
Live trait dials as the headline, knobs to turn them, prompt comparison inside the proof. Effort XL, risk medium. ✅ Does what no prompt can: shows internal state live ✅ Dials and knobs share one engine. ❌ Stops short of the full CRUD story you asked about.
D) Full CRUD (recommended)
C plus a saved trait library (Create) and run-time directional ablation (Delete), all with report cards. Effort XL, risk high. ✅ The clearest paradigm pitch: model behaviors become records you create, read, update and delete ✅ Create and run-time Delete reuse the same directions, so they add little engine code. ❌ Delete is exactly how safety refusals get removed, so it needs a firm safety stance (D2) first.

## Answered: D2 (approved: toggle, guard on by default)
Commitment comparison:

| Commitment | Source/approval or pending | Current | A | B |
|---|---|---|---|---|
| All CRUD operations possible, including refusal Delete | user ask | — | yes, with switch | yes |
| Refusal direction computed per model at load | pending | no | yes | yes |
| Guard state on first run | pending | — | on | off |
| Research-mode switch in UI, no code edits | user ask | — | yes | yes |
| Report card records guard state on every patch | pending | — | yes | yes |
| `presets/llama-refusal.json` | current repo | in UI | research mode only | in UI |

Question: D2 — D2: Full capability with a safety toggle: should the refusal guard start on or off?
Header: D2 Safety
A) Toggle, guard on by default (recommended)
Everything stays possible; one "research mode" switch turns the guard off; patches record the guard state. Effort M, risk low. ✅ Keeps all the reach you want, with no code edits needed ✅ Demo and first run show a tool with a stance, the best answer to "is this a jailbreak tool?". ❌ One extra click before refusal experiments.
B) Toggle, guard off by default
Same switch, but the first run has no guard; turning it on is opt-in. Effort M, risk medium. ✅ Zero friction on first run ✅ Signals a pure frontier research tool. ❌ The first thing a critic can click is "delete refusal", and the toggle then reads as a fig leaf.

## 0E. Mode
Mode: SCOPE EXPANSION; approved decisions: D1 (full CRUD), D2 (guard toggle, on by default). Provenance: explicit user instruction ("Go to CEO review" on a request for scope expansion); no mode question asked.
No new approach decision was needed beyond D1 and D2.

## 0G. Scope expansion analysis

**10x check:** one direction per trait powers all four verbs (Create, Read, Update, Delete). The added effort over the original concept is mostly the Read dial and run-time Delete; both reuse directions already computed.

**Platonic ideal:** pick a model -> its trait fingerprint appears -> type "sycophantic", a new dial appears with its report card (about 80-90 s today; faster Create is a goal) -> chat, dials move per token -> drag a dial down or delete it -> filmstrip shows the dose window -> export a llama.cpp patch and a 10-second clip.

**Delight scan (about 30 minutes each):**
1. Arrow keys nudge the selected dial's dose by 0.05.
2. Hover any word to see every dial's value at that token.
3. "Surprise me" applies a random validated trait from the library.
4. Recipe in the URL hash, so a link restores the exact run.
5. Each trait gets a fixed color used in its dial, knob and filmstrip.

### Expansion proposals (pending, asked one by one)

| ID | Proposal | Effort | Risk |
|---|---|---|---|
| E1 | Starter trait library: 6-8 validated traits shipped with report cards (original proposal; E1 decision narrows it to traits that pass) | M | low |
| E2 | Multi-turn chat mode with live dials (sycophancy rising across turns) | M | medium |
| E3 | Trait fingerprint: same traits measured on two models, side by side | M | medium |
| E4 | `.lobot` patch file: direction + recipe + report card, import/export | M | low |
| E5 | One-click 10-second clip/GIF of dials + filmstrip for X | M | medium |
| E6 | Bake Delete into model weights and export a full GGUF | XL | high |
| E7 | Delight scan items 1-5 as one bundle | S | low |

### Expansion answers (2026-10-29, one answer per proposal)

| ID | Decision | Exact answer |
|---|---|---|
| E1 starter trait library | ACCEPTED | "A) Add (recommended)" |
| E2 multi-turn chat with live dials | ACCEPTED | "A) Add (recommended)" |
| E3 two-model trait fingerprint | ACCEPTED (user overrode the defer recommendation) | "A) Add" |
| E4 `.lobot` patch file | ACCEPTED | "A) Add (recommended)" |
| E5 one-click clip for X | ACCEPTED | "A) Add (recommended)" |
| E6 bake Delete into weights + GGUF | DEFERRED to TODOS.md | "B) Defer (recommended)" |
| E7 delight bundle | ACCEPTED | "A) Add (recommended)" |

## Accepted scope (working plan)

Capabilities, each with a report card (judge + random control + side effects; knob-vs-prompt for Update):

1. **Create:** describe a trait -> contrast prompts (fixed question bank, two instructions) -> direction per layer -> layer chosen by measured effect -> saved trait with report card. Eng D12 (user choice): in wave 1, Create accepts only validated trait families; other descriptions answer "not supported yet". Starter library (E1): up to 8 traits per hero model, only those that pass the gates; none promised in advance.
2. **Read:** live per-token dial = projection of the residual stream on the trait direction; several dials at once; hover a word for all dial values (E7).
3. **Update:** steer with the direction at a dose; batched multi-dose filmstrip streaming live; prompt-only column for the knob-vs-prompt verdict.
4. **Delete:** run-time directional ablation of the trait direction; same report card plus side-effect checks.
5. **Chat mode (E2):** multi-turn, dials per token across turns, steering persists across turns.
6. **Fingerprint (E3):** same trait set on two models via one shared judged benchmark (same prompts, same judge, O7); a trait that fails on a model shows as "unavailable", never hidden; models load one at a time.
7. **Patch (E4):** `.lobot` = zip of safetensors directions + JSON recipe, report card, guard state and execution fingerprint (model revision, tokenizer, template, dtype, device) (S3-1, O8); never unpickled; Update also exports as a llama.cpp control vector, re-checked against lobot output.
8. **Clip (E5):** one-click 10-second phone-sized clip of dials + filmstrip.
9. **Refusal guard (D2 as amended by O5):** on by default; before any run, the whole set of active edits is checked on a fixed refusal prompt set; a set that weakens refusals beyond a threshold is blocked; research-mode switch turns it off; patches record guard state; refusal preset in research mode only; enforced from wave 1.
10. **Delight (E7):** arrow-key dose nudge, hover values, Surprise me, recipe in URL hash, fixed trait colors.
11. **Click a word (R1):** list the top neurons that fired at a token; push one as a single-neuron Update.

Hero model: Llama 3.2 3B. Single-neuron mode stays as an advanced view.

## NOT in scope

- E6 bake Delete into weights + full GGUF export: DEFERRED (XL, multi-GB files, high safety exposure; run-time Delete covers the demo).
- Teaching new knowledge or skills: not possible with steering (stated limit).
- Hosted or multi-user deployment: local only (constraint).

## Spec review round 1 (2026-10-29): score 5/10, FAIL; corrections applied

Corrections (no scope change):
- Evidence restated: pre-registered verdicts 1 clean / 1 weak / 4 none; the earlier "5/6 behaviors work" claim used peak-dose numbers and is unverified.
- Create time: about 80-90 s per trait today [measured: spike v2]; "a dial in seconds" is a goal, not a fact.
- Implementation owner must define and prove: (a) fluency metric replacing NLL (for example a local fluency judge plus a degenerate-repetition check); (b) guard measure = cosine overlap between trait and refusal directions at the chosen layer, threshold set from a calibration of known-safe vs refusal traits; (c) dial scale = projection normalized by the trait's contrast-set mean gap (0 = neutral mean, 1 = trait mean); (d) knob-vs-prompt win rule = higher judge score at equal or better fluency on held-out prompts; (e) Delete test = ablate only on prompts that elicit the trait at baseline, report trait drop and side effects.
- Edge cases: imported patch with guard off while local guard is on -> refuse apply until research mode, show why; trait with no layer passing the gate -> trait saved as "no knob found", not usable; chat beyond token budget -> truncate oldest turns, dials keep per-turn history.
- E1: ship only traits that pass the gates, up to 8; none promised in advance.
- E3: models load one at a time (Llama 3.2 3B, then Qwen 3.5 4B), results cached; never both in memory.
- Single-neuron mode stays as an advanced view (now also in the CEO summary).

Review decisions (2026-10-29):
- R1 click-a-word neuron discovery: ACCEPTED, user answer "A) Add" (overrode the defer recommendation). Placed in wave 2 with Read. Implementation owner must prove: top neurons by activation at the clicked token, shown with an honest "neurons often mix several jobs" note; one click pushes a neuron as a single-neuron Update.
- R2 delivery order: ACCEPTED, user answer "A) Waves with gates (recommended)".

## Delivery waves (approved R2)

| Wave | Contents | Gate to start the next wave |
|---|---|---|
| 0 | Gates + ONE complete trait end to end (O2): (1) Read gate judged on behavior, natural and steered text, incl. projection-moves-behavior-does-not cases (O1); (2) context-aware judge validated on hand-labeled examples (O3); (3) three data splits, equal tuning budget for prompt baseline (O4); (4) batched filmstrip memory and time on 24 GB, with chat cache semantics decided first (O6); (5) llama.cpp control-vector round-trip vs lobot output (O8) | Each gate reports pass/fail on the final untouched split; a failed gate removes or reshapes its dependent item, with a user decision |
| 1 | Create (trait + per-verb eligibility + report card + starter library of passing traits), Update (vector steering), live batched filmstrip, knob-vs-prompt verdict, refusal guard enforced (behavioral check on all active edits, O5; D2 default on + research switch) | Report cards on the final split for every shipped trait; guard check passes |
| 2 | Read dials (several, hover values), click-a-word neuron discovery (R1), chat mode (E2) | Read gate result reproduced in the UI |
| 3 | Delete (run-time ablation, only for traits eligible for Delete) | Delete report cards with side effects, on prompts that elicit the trait; guard re-checked with Delete active |
| 4 | `.lobot` patch + llama.cpp export (E4), clip (E5), two-model fingerprint (E3), delight bundle (E7) | — |


## 0I. Temporal interrogation (feasibility)

```
  HOUR 1 (foundations):  promote spike code into an engine module; trait object = name, model, layer, direction, scale, report card
  HOUR 2-3 (core logic): batched per-row dose hook; projection read-out per token; ablation hook; which layer per verb?
  HOUR 4-5 (integration):stream dials + filmstrip over NDJSON; chat KV cache with hooks; guard check before Delete/negative Update
  HOUR 6+ (polish/tests):report cards on held-out prompts; patch round-trip; llama.cpp control-vector round-trip; clip capture
```
E3 memory: load Llama 3.2 3B, then Qwen3.5-4B, never both in memory; cache each model's trait results (about 6.4 GB and 8-9 GB, each fits 24 GB alone).
Feasibility blockers (must pass before the matching wave): Read gate (dial tracks trait on held-out inputs, incl. user-induced sycophancy); batched filmstrip memory + time on 24 GB; llama.cpp control-vector round-trip (llama.cpp not installed here).
Effort: human team about 4-6 weeks; CC + gstack about 3-5 focused days [estimated].

## Spec review result
3 reviewer launches (cap). Scores: 5/10 (round 1), 8/10 (round 2), 9/10 (round 3, latest). Round 3 found 1 stale line ("dial appears in seconds" in 0G); fixed after the cap. issues verified: 1 of 1 (applied edit checked by author; spec not re-verified by the reviewer).

# Review sections (strategy-only depth)

Current scope: SCOPE EXPANSION (explicit). Accepted: D1, D2, E1, E2, E3, E4, E5, E7, R1, R2. Deferred: E6. Rejected: none. Pending: none.

## Section 1: Architecture

```
                         +---------------------------- web/index.html ----------------------------+
                         | Describe | Dials (Read) | Filmstrip (Update) | Delete | Chat | Clip | ... |
                         +--------------------------------+------------------------------------------+
                                                          | NDJSON streams + JSON
                                                   +------v------+
                                                   |  server.py  |  127.0.0.1 only
                                                   +------+------+
                         +--------------------------------+---------------------------------+
                         |                                |                                 |
                  +------v-------+                 +------v-------+                  +------v-------+
                  | traits.py    |  NEW            | steer.py     |  EXISTING        | patches.py   | NEW
                  | create: contrast,|             | hooks: neuron add |             | .lobot import/export,
                  | direction, layer,|             | + NEW: vector add,|             | llama.cpp control vector
                  | report card,     |             | batched dose,     |             +--------------+
                  | guard check      |             | projection read,  |
                  +------+-------+                 | ablation          |
                         |                         +------+-------+
                  +------v-------+                        |
                  | traits/ store|  NEW (local files)     |  HF transformers, MPS
                  +--------------+                        v
                                                   model (one in memory)
```

Data flows (Create, the new main flow):
```
  happy: description -> template contrast sets -> capture per layer -> direction -> per-layer gate -> trait saved + report card
  nil:   no description -> 400 "Describe a trait first."
  empty: description of only spaces/1 char -> 400; description the model cannot contrast (no layer passes) -> trait saved as "no knob found", unusable
  error: OOM / MPS error during capture -> stage "error" with cause; no partial trait written (write after all gates)
```

State machine: trait
```
  draft --create--> validated --> per-verb eligibility (O2), each judged on the final split:
                                    read: yes/no    update_up: yes/no    update_down: yes/no    delete: yes/no
    no verb eligible -> no_knob (terminal, report kept)
  run time: active edit set --guard (O5, behavioral refusal check)--> allowed | blocked (until research mode)
  invalid: an ineligible verb is never offered; edits to a trait create a new version
```

Findings:
1. WARNING: engine state lives in `app.py` (Gradio file) and `server.py` imports it as the engine (`server.py:32`). CRUD adds four hook kinds and a trait store; leaving them in a UI file is the "poor pattern" from taste calibration. Implementation owner must prove: engine state moves to an engine module before wave 1, Gradio becomes a thin client or is retired.
2. WARNING: hook composition. Update (add), Read (projection, read-only), Delete (ablation) and neuron rows can be active at once. Order matters (ablate then add vs add then ablate). Owner must define and test one fixed order.
3. OK: single model in memory (E3 sequential) keeps the single point of failure the same as today: the warm model.
4. Scaling: local single user; 10x = 10 dials and 7 doses at once. First break: batch memory (wave-0 gate 2).
5. Rollback: git revert per wave; trait store and patches are new files, so rollback leaves old UI working.
6. Platform potential: `traits/` + `.lobot` are the platform; a later hub only needs to share these files.

## Section 2: Error & Rescue map (capability level)

```
  CAPABILITY               | WHAT CAN GO WRONG                          | CLASS (to be named by owner)
  -------------------------|--------------------------------------------|------------------------------
  Create                   | OOM on capture; no layer passes; judge     | MemoryError/MPS error; NoKnob; JudgeError
                           | returns no Yes/No mass                     |
  Read dial                | direction shape mismatch after model swap  | TraitModelMismatch
  Update / filmstrip       | batch OOM; Stop mid-batch                  | MemoryError; cancelled
  Delete                   | ablation collapses fluency                 | (not an exception: report card fail)
  Chat                     | context past token budget                  | ContextOverflow
  Patch import             | wrong model, wrong dims, corrupted file,   | PatchModelMismatch; PatchFormatError;
                           | guard-off patch while guard on             | GuardBlocked
  llama.cpp export         | layer index off by one; unsupported model  | ExportVerifyError
  Clip                     | browser recording API missing              | (UI fallback)

  CLASS              | RESCUED? | RESCUE ACTION                                 | USER SEES
  -------------------|----------|-----------------------------------------------|------------------------------
  MemoryError/MPS    | Y        | stop, free cache, stage "error"               | "Out of memory: try fewer doses"
  NoKnob             | Y        | save as no_knob with its report               | "No usable knob found for X"
  JudgeError         | Y        | mark report card metric "unavailable"         | metric shown as unavailable, never as pass
  TraitModelMismatch | Y        | refuse to load trait                          | "Trait built for <model>"
  ContextOverflow    | Y        | drop oldest turns, keep dial history          | notice "older turns trimmed"
  PatchFormatError   | Y        | refuse import                                 | "Not a valid .lobot file"
  GuardBlocked       | Y        | refuse apply, explain switch                  | "Guard is on: this patch weakens refusals"
  ExportVerifyError  | Y        | no file written                               | "Export check failed: <cause>"
```
GAP (owner must close): the current server catches `Exception` in `_run_stream` (`server.py`, broad `except` with a noqa). Acceptable for v1 surfacing, but new capabilities must name their classes above and log args (trait, model, layer, dose).

## Section 3: Security & threat model

| Threat | Likelihood | Impact | Mitigated? |
|---|---|---|---|
| T1 `.lobot` import runs code: a pickled tensor (`torch.load`) from a shared patch executes arbitrary code | High (patches are meant to be shared) | High | YES: S3-1 approved safetensors + JSON in a zip; imports never unpickle; hostile-QA test rejects a pickled payload |
| T2 API reachable from other hosts | Low | High | yes: `uvicorn.run(host="127.0.0.1")` (`server.py`) |
| T3 malicious web page calls localhost API (CSRF-style POST to 127.0.0.1:7860) | Medium | Medium (runs generations, imports patches) | partial: JSON content type; owner must prove an Origin check on mutating routes |
| T4 trait description as prompt injection into templates | Medium | Low (local model, local user) | owner: description is data in a fixed template, length-capped |
| T5 refusal removal | — | — | D2 guard on by default (approved) |
| T6 path traversal in trait/patch names | Medium | Medium | owner must prove: names slugged, store dir fixed |

## Section 4: Data flow & interaction edge cases

```
  PATCH IMPORT: file -> size cap -> parse (no code exec) -> schema/version -> model id + dims match -> guard check -> store -> UI
     shadows: empty file -> PatchFormatError | huge file -> size cap | wrong model -> mismatch | guard-off patch -> GuardBlocked | duplicate name -> new version
  READ DIAL: token hidden state -> projection -> normalize (contrast gap) -> stream -> UI
     shadows: trait for other model -> mismatch | NaN (fp16 overflow) -> show "n/a", log | no traits selected -> dials panel empty state
```
Async ordering: one run lock (existing `engine._run_lock`) covers generations; new flows that mutate the trait store (create, import, delete) must take the same lock or a store lock. Invariant: a run never sees a trait half-written. Mechanism: write to temp file, atomic rename. Owner must test: create-during-run and import-during-run orders.

| Interaction | Edge case | Handled? | How |
|---|---|---|---|
| Create | double click | owner | disable button while running (pattern exists for Run) |
| Create | navigate away mid-create | owner | server finishes, trait appears next load |
| Filmstrip | Stop mid-batch | yes | CancelCriteria per token (existing) |
| Dials | 0 traits / 20 traits | owner | empty state; cap visible dials at 6, list the rest |
| Chat | very long chat | owner | ContextOverflow rescue |
| Guard switch | toggled during a run | owner | applies to next run only |

## Section 5: Code quality
1. Spike code (`scripts/spike_knob.py`) must be promoted, not copied: capture, add_vector, judge become engine functions; the script imports them.
2. `scripts/derive_llama_recipe.py:capture_means` and spike `capture_last` overlap (both hook every block). One capture helper.
3. Naming: "dose" today means a neuron scale (`steer.py`) and a residual-norm fraction (spike). Owner must use one unit per mode and label both in the UI and recipe.
4. Over-engineering risk: E3 fingerprint and E5 clip are wave 4; no abstraction for them before then.

## Section 6: Test review

```
  NEW                         | TYPE           | HAPPY                         | FAILURE                          | EDGE
  ----------------------------|----------------|-------------------------------|----------------------------------|------------------------
  Create (trait + report)     | unit (stub) +  | trait saved with report       | no layer passes -> no_knob       | empty description
                              | model lane     |                               |                                  |
  Read dial normalization     | unit           | 0 at neutral mean, 1 at trait | NaN -> n/a                       | zero contrast gap
  Batched dose hook           | unit + model   | row i gets dose i             | OOM -> error stage               | 1 dose, 7 doses
  Ablation                    | unit + model   | component removed (dot ~ 0)   | fluency fail -> report card fail | trait absent at baseline
  Guard                       | unit           | overlap > threshold blocks    | research mode allows + labels    | threshold boundary
  Patch round-trip            | unit           | export -> import identical    | corrupted, wrong model, pickle   | guard-off patch
  llama.cpp export            | integration    | same top tokens as lobot      | layer off-by-one detected        | (wave-0 gate 3)
  Chat                        | model lane     | dial history per turn         | overflow trims                   | 1 turn
```
- Every behavior asserts an observable result; `no_claim` for stubbed tests: "real model behavior (engine stubbed; unit only)", as `tests/test_server.py` does today.
- 2am-Friday test: a report card that says PASS must reproduce on held-out prompts in the model lane.
- Hostile QA: import a `.lobot` with a pickled payload; toggle guard mid-run; create two traits with the same name at once.
- Chaos: kill the server mid-create; restart; store has no half trait.
- Flakiness: model-lane tests are greedy and seeded (deterministic, proven bit-exact); judge-based report cards are not unit tests.

## Section 7: Performance
- Slowest paths: Create (82-96 s per trait [measured: spike v2]); filmstrip batch [estimated, wave-0 gate]; report card generation (judge passes).
- Caching: per model, trait directions and report cards are files; starter library ships precomputed; calibration medians already cached.
- Memory: one model (6.4 GB Llama / 8-9 GB Qwen) + batch KV cache for up to 8 rows x 256 tokens: small vs 24 GB, but MPS fragmentation is the unknown (gate 2).
- Read dials: one dot product per token per trait: negligible.

## Section 8: Observability
- Each run logs one structured line: run id, model, verbs used (traits, layers, doses), guard state, timings per stage, outcome.
- Report cards are the product metric: stored with the trait.
- Debuggability: a bug report needs the recipe string + trait files; both exist by design.
- Joy to operate: a local `/api/health` with model, memory in use, last error.

## Section 9: Deployment & rollout
- Local app, no migrations. Rollout = waves; each wave is one PR-sized change (default one PR per wave).
- Feature flags: research mode is the only runtime switch (D2). No other flags.
- Post-change check per wave: unit suite, model lane for the new verb, live run in the browser (as done for Stop).
- Rollback: revert the wave's commit; trait store files are additive.

## Section 10: Long-term trajectory
- Debt: spike code to promote; engine-in-app.py; dose units.
- Reversibility: 4/5 (all additive; `.lobot` format is the one-way door once shared: version it from v1).
- Phase 2: E6 baked Delete; a shared patch hub; multi-model traits. Architecture supports it if traits and patches stay model-tagged files.

## Section 11: Design & UX

```
  first run -> [Model fingerprint: starter dials] -> pick/describe trait -> [Create progress, 80-90 s, stages]
     -> [Chat or prompt] -> dials stream per token -> click word: top neurons
     -> drag dial (Update) -> filmstrip streams + prompt column + verdict
     -> Delete -> report card (side effects) -> Export patch / Clip
  guard on + refusal-like trait -> [blocked state with "research mode" switch]
```

| Feature | Loading | Empty | Error | Success | Partial |
|---|---|---|---|---|---|
| Create | staged progress, 10-20 min [estimated, OV8] | no traits: starter library | OOM / no knob | trait + report card | report metric unavailable |
| Dials | per-token stream | "select a trait" | n/a value | live dials | some traits mismatched to model |
| Filmstrip | columns stream | — | OOM, stopped | all doses + verdict | stopped mid-batch |
| Delete | progress | — | guard blocked | report card | fluency fail flagged |
| Patch | — | — | invalid/mismatch/guard | imported | — |

- DESIGN.md alignment: pink stays reserved for the split; dials need their own color rule (E7 fixed trait colors) without breaking "pink only at the split". Owner: add a trait palette to DESIGN.md.
- Create takes 10-20 min [estimated, OV8; corrected by design review]: see design decision 3.
- Recommend /plan-design-review before wave 1 UI work.

## Answered: S3-1 (approved: A, safetensors + JSON)
Commitment comparison:

| Commitment | Source/approval or pending | Current | A | B | C |
|---|---|---|---|---|---|
| `.lobot` patch import/export (E4) | approved E4 | planned | yes | yes | yes |
| Tensor storage | pending | unspecified | safetensors | torch file, `weights_only=True` | unspecified |
| Metadata storage | pending | unspecified | JSON | inside torch file | unspecified |
| Code execution possible on import | Section 3 T1 | possible | no (format has no code) | reduced, depends on torch loader | possible |
| Tests: pickled payload rejected | Section 6 hostile QA | pending | yes | yes | no |

Question: D3 — S3-1: How should `.lobot` patch files store their data, given patches are made to be shared?
Header: S3-1 Patch
A) safetensors + JSON (recommended)
Directions in safetensors, recipe and report card in JSON, in one zip; imports never unpickle. Effort S, risk low. ✅ A shared patch cannot run code on import, by format design ✅ safetensors is already a project dependency. ❌ A slightly more involved writer than a single torch.save call.
B) torch file with weights_only
One torch.save file loaded with weights_only=True. Effort S, risk medium. ✅ Simplest to write ✅ The PyTorch loader blocks most code. ❌ Safety depends on the loader and its version, not on the format.
C) Leave it to implementation
No rule now. Effort S, risk high. ✅ No decision cost today ✅ Keeps options open. ❌ The default torch.save/torch.load path can run code from a downloaded patch.

## Outside voice (Codex, provider=codex, model gpt-6-astra per probe; completed)

Verdict: findings (P1). Recommendation: "revise wave 0 around one independently validated trait with explicit per-operation eligibility".
Author verification: citations checked: `app.py:97` (revision fetched, not passed to loaders), `scripts/spike_knob.py:306` (judge sees only reply + label), `scripts/spike_knob.py:359` (held-out set used for layer choice; best dose chosen on the reported prompts). All confirmed.

| Row | Finding (severity) | Proposed remedy | Status |
|---|---|---|---|
| O1 | Read gate can validate itself: projection of the added direction rises by construction (High) | Read gate uses independently judged behavior on natural and steered generations, incl. projection-moves-but-behavior-does-not cases | approved (D4: "A) Apply (recommended)") |
| O2 | One direction does not prove four verbs (High) | Per-verb eligibility per trait (Read, Update+, Update-, Delete); wave 0 proves one complete trait end to end before the general CRUD layer | approved (D5: "A) Apply (recommended)") |
| O3 | Judge lacks question context; cannot judge sycophancy or factual drift (High) | Judge sees question + reply; rubric validated on hand-labeled examples before its scores count as proof | approved (D6: "A) Apply (recommended)") |
| O4 | "Held out" is tuning data (High) | Three splits: extraction, tuning (layer/dose/threshold), final untouched test; prompt baseline gets the same tuning budget | approved (D7: "A) Apply (recommended)") |
| O5 | Guard by cosine does not prove refusals survive; positive Update and combinations escape; guard lands after Update ships (High) | Guard = behavioral refusal check on a fixed refusal set for every active intervention set; enforced from wave 1 | approved (D8: "A) Apply (recommended)") |
| O6 | Chat cache semantics undefined (Medium) | Pending implementation choice, owner: wave-2 lead, decide before the wave-0 performance gate: edits apply to future tokens only (default) vs recompute context | pending, non-blocking, owner named |
| O7 | Fingerprint scale not comparable; "only passing traits" is selection bias (Medium) | Shared behavioral benchmark (judge on same prompts) for both models; failures shown as "unavailable", never hidden | approved (D9: "A) Apply (recommended)") |
| O8 | Patch compatibility weaker than replay promise (Medium) | Bind traits, patches and report cards to an execution fingerprint (model revision, tokenizer, template, dtype, device); pass revision to loaders; re-validate llama.cpp export | approved (D10: "A) Apply (recommended)") |

Approval readiness: PASS. Checked rows and answers: D1 ("D) Full CRUD"), D2 ("A) Guard on by default"), E1-E5/E7 (per-proposal answers), E6 (deferred), R1 ("A) Add"), R2 ("A) Waves with gates"), S3-1 ("A) safetensors + JSON"), O1-O5, O7, O8 ("A) Apply" each, D4-D10). O6 stays pending with an owner (non-blocking). No accepted work lacks an answer.

# Required outputs

## Reality rows

REALITY: premise-table finding unverified=3 (llama.cpp control-vector format, batched filmstrip in 24 GB, 5/6-knob claim) logs/spike_knob/v2/report.md:5
REALITY: already-done pass no commits for vectors, filmstrip, dials or export; spike script uncommitted scripts/spike_knob.py:1
REALITY: surface-check finding llama-cli and llama-server not installed; control-vector flags unverified (wave-0 gate) server.py:1
REALITY: binding-decisions pass gstack-decision-search empty; no plan binding section overruled CLAUDE.md:1
REALITY: numbers pass every number tagged measured or estimated; batched filmstrip time marked re-measure before gate logs/spike_knob/v2/report.md:5
REALITY: deferred-asks pass no linked issue TODOS.md:1
REALITY: pr-count pass no .gstack/ship-policy.json; default one PR per wave pyproject.toml:1

## What already exists
See 0B table: spike capture/direction/judge/random control (`scripts/spike_knob.py`), contrast capture (`scripts/derive_llama_recipe.py`), neuron hooks and calibration (`steer.py`), stream + Stop (`server.py`, `steer.CancelCriteria`), recipe string (`steer.recipe_string`), UI + DESIGN.md (`web/`). All reused; engine state moves out of `app.py` (Section 1 finding 1).

## Dream state delta
After waves 0-4: Create/Read/Update/Delete with per-verb eligibility and honest report cards, chat dials, fingerprint, patches and clips, on one laptop. Still missing vs the 12-month ideal: baked Delete (E6), a shared patch hub, faster Create (target: seconds; today 80-90 s).

## Scope expansion decisions
Accepted: D1, D2 (amended by O5), E1, E2, E3 (amended by O7), E4 (S3-1, O8), E5, E7, R1, R2, O1-O5, O7, O8. Deferred: E6 (TODOS.md). Skipped: none. Pending: O6 (owner named).

## Error & rescue registry (capability level)
See Section 2 tables: 9 capability rows, 8 named rescue classes. Owner must verify each class before its wave; no unknown is assumed safe.

## Failure modes registry

```
  CAPABILITY      | FAILURE MODE                         | RESCUED? | TEST?   | USER SEES?                       | LOGGED?
  ----------------|--------------------------------------|----------|---------|----------------------------------|--------
  Create          | no verb eligible                     | Y        | planned | "No usable knob found"           | Y
  Create          | OOM during capture                   | Y        | planned | "Out of memory"                  | Y
  Read            | dial moves, behavior does not (O1)   | Y (gate) | planned | dial not offered for that trait  | Y
  Report card     | judge scores w/o context (O3)        | Y (gate) | planned | metric "unavailable"             | Y
  Report card     | selection leakage (O4)               | Y        | planned | pass only on final split         | Y
  Update/batch    | OOM / Stop                           | Y        | partial | error / stopped stage            | Y
  Delete          | fluency collapse                     | Y        | planned | report card fail                 | Y
  Guard (O5)      | combined edits weaken refusals       | Y        | planned | "Guard is on: blocked" + switch  | Y
  Patch import    | code in file (T1)                    | Y (S3-1) | planned | "Not a valid .lobot file"        | Y
  Patch import    | wrong revision/tokenizer (O8)        | Y        | planned | "Patch built for <fingerprint>"  | Y
  Chat            | cache semantics (O6)                 | unknown  | unknown | unknown (owner decides)          | -
  Export          | llama.cpp layer off-by-one           | Y (gate) | planned | "Export check failed"            | Y
  Fingerprint     | trait fails on one model (O7)        | Y        | planned | "unavailable"                    | Y
```
Critical gaps (RESCUED=N, TEST=N, silent): 0. O6 is unknown, not silent; it has an owner and a deadline (before the wave-0 performance gate).

## Diagrams produced
System architecture (Section 1), Create data flow with nil/empty/error paths (Section 1), trait state machine with per-verb eligibility (Section 1), patch import and read-dial data flows (Section 4), error flow (Section 2 tables), user flow (Section 11), 0C dream state, 0I temporal map. Deployment and rollback: local app, revert per wave (Section 9), no separate diagram needed.

## Stale diagram audit
Files this plan touches: `steer.py` docstring (load recipe block: still accurate), `server.py` docstring (endpoint list: will need new routes per wave), `DESIGN.md` (needs a trait palette, Section 11). No stale diagrams today.

## Implementation tasks (strategy-only: next research, design or verification actions)

- [ ] **T1 (P1, human: ~1 day / CC: ~2h)** — engine — Move engine state out of `app.py` into an engine module
  - Surfaced by: Section 1 finding 1
  - Files: `app.py`, `server.py`, new engine module (to be determined)
  - Verify: `uv run python -m unittest discover -s tests -t .` green; live run matches the recorded artifact
- [ ] **T2 (P1, human: ~2 days / CC: ~3h)** — eval — Build three-split evaluation and a context-aware judge validated on hand labels
  - Surfaced by: Outside voice O3, O4
  - Files: `scripts/spike_knob.py` (promote), new eval module (to be determined)
  - Verify: judge agreement with hand labels reported; final split untouched until the end
- [ ] **T3 (P1, human: ~2 days / CC: ~3h)** — wave 0 — Prove one complete trait end to end with per-verb eligibility
  - Surfaced by: Outside voice O1, O2
  - Files: to be determined
  - Verify: Read/Update/Delete eligibility judged on the final split; negative Read cases included
- [ ] **T4 (P1, human: ~1 day / CC: ~1h)** — wave 0 — Measure batched filmstrip memory and time on 24 GB; decide chat cache semantics first
  - Surfaced by: 0I gate, O6
  - Files: to be determined
  - Verify: peak memory and wall time logged for 1, 4 and 7 doses
- [ ] **T5 (P1, human: ~1 day / CC: ~1h)** — export — llama.cpp control-vector round-trip
  - Surfaced by: reality surface-check, O8
  - Files: to be determined
  - Verify: same top tokens in llama.cpp and lobot for one trait and dose
- [ ] **T6 (P1, human: ~1 day / CC: ~2h)** — guard — Behavioral refusal check on active edit sets, enforced from wave 1
  - Surfaced by: O5, D2
  - Files: to be determined
  - Verify: known refusal-weakening sets blocked; harmless sets allowed; research mode labels patches
- [ ] **T7 (P2, human: ~2h / CC: ~20min)** — replay — Pin model revision in loaders; add execution fingerprint to recipe
  - Surfaced by: O8 (`app.py:97`)
  - Files: `app.py`, `steer.py`
  - Verify: fingerprint recorded; a patch with a different revision is refused
- [ ] **T8 (P2, human: ~1h / CC: ~10min)** — security — Origin check on mutating localhost routes
  - Surfaced by: Section 3 T3
  - Files: `server.py`
  - Verify: cross-origin POST rejected in a test
- [ ] **T9 (P3, human: ~2h / CC: ~20min)** — design — Add a trait palette to DESIGN.md; run /plan-design-review before wave 1 UI
  - Surfaced by: Section 11
  - Files: `DESIGN.md`
  - Verify: palette passes contrast checks; pink stays split-only

## Completion summary

```
  +====================================================================+
  |            MEGA PLAN REVIEW — COMPLETION SUMMARY                   |
  +====================================================================+
  | Mode selected        | SCOPE EXPANSION                             |
  | System Audit         | engine state in app.py; spike evidence weak |
  |                      | (1 clean/1 weak/4 none); no llama.cpp here  |
  | Step 0               | D1 full CRUD, D2 guard on by default        |
  | Section 1  (Arch)    | 2 issues found                              |
  | Section 2  (Errors)  | 9 error paths mapped, 1 GAP (broad except)  |
  | Section 3  (Security)| 5 issues found, 1 High severity (resolved)  |
  | Section 4  (Data/UX) | 6 edge cases mapped, 0 unhandled (owners)   |
  | Section 5  (Quality) | 4 issues found                              |
  | Section 6  (Tests)   | Diagram produced, 0 gaps (all planned)      |
  | Section 7  (Perf)    | 2 issues found (Create time, batch memory)  |
  | Section 8  (Observ)  | 1 gap found (structured run log)            |
  | Section 9  (Deploy)  | 1 risk flagged (.lobot format one-way door) |
  | Section 10 (Future)  | Reversibility: 4/5, debt items: 3           |
  | Section 11 (Design)  | 2 issues (trait palette, Create progress)   |
  +--------------------------------------------------------------------+
  | NOT in scope         | written (3 items)                           |
  | What already exists  | written                                     |
  | Dream state delta    | written                                     |
  | Error/rescue registry| 9 rows, 0 CRITICAL GAPS                     |
  | Failure modes        | 13 total, 0 CRITICAL GAPS                   |
  | TODOS.md updates     | 1 item (E6)                                 |
  | Scope proposals      | 8 proposed, 7 accepted (E1-E5, E7, R1)      |
  | CEO plan             | written                                     |
  | Outside voice        | codex, completed (8 findings, 7 approved)   |
  | Lake Score           | 8/8 recommendations chose complete option   |
  | Diagrams produced    | 8 (arch, flows, state, error, user, dream,  |
  |                      | temporal)                                   |
  | Stale diagrams found | 0                                           |
  | Unresolved decisions | 1 (O6, pending with owner)                  |
  | Reality rows         | 7/7 rows, 3 unverified                      |
  +====================================================================+
```

### Unresolved decisions
- O6 chat cache semantics: future tokens only vs recompute. Owner: wave-2 lead; decide before the wave-0 performance gate.

# Eng review (/plan-eng-review, 2026-10-29)

Target: this plan, waves 0-1 in focus. Report file: this file. Design doc: `docs/plans/behavior-crud.md` (promoted CEO plan).
Prior learnings applied: steering-gates-self-validate (8/10, 2026-10-29); bpe-split-offset-leading-space (9/10, 2026-10-29).

## Scope challenge

Complexity (waves 0-1) [estimated]: new modules engine, traits, evaluation, guard (4 new classes: Engine, Trait, Evaluator/Judge, Guard); edits steer.py, server.py, app.py, web/index.html, scripts/spike_knob.py, README.md; new tests for engine, traits, evaluation, guard; edit tests/test_server.py. About 15 files. Gate B trips (8+ files, 2+ new classes).

Findings:
1. [CRITICAL] (confidence: 8/10) app.py:58 `_model = None` + steer.py:237 `_active_steering: list = []` — engine state has two owners; spike `add_vector` registers its own forward hook outside `_active_steering`, so `steering_hook_count() != 0` guards (steer.py:332, :518) cannot see vector hooks; calibration or judging could run with a vector hook active and corrupt medians or scores silently.
2. [INFORMATIONAL] (confidence: 9/10) steer.py:281 `x[..., n] = x[..., n] + d` — one scalar per neuron for every batch row; a batched multi-dose filmstrip needs a per-row dose.
3. [CRITICAL] (confidence: 6/10, medium confidence, verify) regression risk: README "The app runs greedily with a fixed seed. Results stay reproducible." Batched or padded greedy decoding can differ from batch size 1, more in fp16 (forum and MLX reports; no direct transformers-on-MPS report found). Recorded artifacts reproduce exactly today (live check 2026-10-29).

### Structure record (B)

feature answers: none proposed (no cuts); structure: B, D1 answer "B) Smaller arrangement (recommended)"; accepted scope: engine.py (Engine), traits.py (Trait + store), evaluation.py (splits, judge, report cards, guard check), all approved CEO features unchanged; pending remedies: E-SC1, E-SC3.

## Decision ledger (eng)

### E-S1: file/class arrangement for waves 0-1
Finding: Scope challenge complexity; reviewer: Claude
Plan baseline: CEO plan names engine module, traits store, eval module, guard (no file arrangement approved)
Runtime evidence: engine code today: steer.py (hooks, calibration), app.py (model state), server.py (API)
Question D1: structure selector (no grid required)
State: approved
Actual answer: B, "B) Smaller arrangement (recommended)"
Accepted scope: three new modules engine.py, traits.py, evaluation.py; guard check lives in evaluation.py

### E-SC1: one owner for engine state and every hook
Finding: Scope challenge 1, CRITICAL, 8/10, app.py:58 + steer.py:237 + spike add_vector; reviewer: Claude
Plan baseline: CEO T1 "Move engine state out of app.py into an engine module" (owner task, mechanism not chosen)
Runtime evidence: model state in app.py globals; neuron hooks in steer._active_steering; spike vector hooks unregistered
Comparison grid:
| Choice | Current | A | B |
|---|---|---|---|
| Owner of model, tokenizer, lock | app.py globals | Engine instance in engine.py | engine.py module globals |
| Owner of hook registry | steer._active_steering (neuron only) | Engine, all hook kinds (neuron add, vector add, ablation, read) | engine.py list, all kinds |
| "No hooks" check before calibration/judge | steering_hook_count() (neuron only) | Engine.clean() context, asserts zero hooks of any kind | same assertion over module list |
| Testability | patch module globals | inject a fake Engine | patch module globals |
| Tests | — | registry counts every kind; clean() raises with any hook active | same |
Question D2:
D2 — E-SC1: Who should own the model state and every steering hook?
State: approved
Actual answer: A, "A) Engine class owns all (recommended)"
Accepted scope: Engine instance in engine.py owns model, tokenizer, lock and one registry of every hook kind (neuron add, vector add, ablation, read); Engine.clean() asserts zero hooks of any kind before calibration and judging; tests: registry counts every kind, clean() raises with any hook active

### E-SC3: reproducibility contract for batched runs
Finding: Scope challenge 3, CRITICAL, 6/10 (verify), README reproducibility claim; reviewer: Claude
Plan baseline: wave-0 gate 4 measures batched filmstrip memory and time (approved R2/O6); no exactness requirement
Runtime evidence: single runs reproduce artifacts exactly (live 2026-10-29); batched equality unknown on MPS
Comparison grid:
| Choice | Current | A | B | C |
|---|---|---|---|---|
| Wave-0 check | memory + time | + token-id equality, batched vs single, every dose | + record batch layout | + run batches in fp32 |
| If unequal | — | filmstrip falls back to sequential single runs (exact, slower) | accept; replay rebuilds the same batch | fp32 (2x memory) |
| README claim | "Results stay reproducible" | kept as is | narrowed: reproducible within the same batch layout | kept, cost 2x memory |
| Regression test | none | model-lane test: batched == single ids | recipe carries batch layout | fp32 equality test |
Question D3:
D3 — E-SC3: What should happen if batched runs do not match single runs token for token?
State: approved
Actual answer: A, "A) Exactness gate + fallback (recommended)"
Accepted scope: wave-0 check compares batched vs single token ids for every dose; any mismatch makes the filmstrip run doses sequentially; model-lane regression test batched == single ids; README reproducibility claim kept

Scope Challenge result: scope accepted as-is.

## Section 1: Architecture

Design-twice: shape A = thin Engine (model + hooks) plus separate Run/Session objects holding edits; shape B = `Engine.run(edit_set)` with an immutable `EditSet` (typed edits: NeuronEdit, VectorEdit, Ablation, Read) and the Engine composing hooks internally. Pick: B, deeper module (hides hook composition, ordering, cleanup and the clean() check behind one call). Red flags: #5 split ownership (fixed by D2), #6 two hook paths (fixed by D2), #2 information leakage: "dose" means a neuron scale (steer.py:259 docstring) and a residual-norm fraction (spike); typed edits carry their unit (required for the approved E4/O8 recipe grammar, no new choice).

```
  server.py --EditSet--> Engine.run() --> per layer: ONE composite hook --> model (MPS)
                              |                 order: ? (E-A1)
                              +--> Engine.clean() (no hooks) --> evaluation.py judge / calibration
  traits.py (Trait: directions, eligibility, report card) <--> traits/ store (safetensors + JSON)
```

Findings:
1. [INFORMATIONAL] (confidence: 8/10) steer.py:286 `register_forward_pre_hook(_prehook)` per layer and spike `register_forward_hook` on the block: with several edit kinds active, the result depends on registration order (pre-hook on down_proj vs forward hook on block). Ablation then add differs from add then ablation.
2. [INFORMATIONAL] (confidence: 7/10) server.py `@api.post("/api/stop")` takes no body, so any web page in the browser can POST it cross-site (a simple request skips the CORS preflight); later multipart patch import (wave 4) has the same exposure. No Origin check exists.

### E-A1: order of edits inside one layer
Finding: Section 1 finding 1, INFORMATIONAL, 8/10; reviewer: Claude
Plan baseline: CEO Section 1 finding 2 "Owner must define and test one fixed order" (order not chosen)
Runtime evidence: unknown for combinations; today only one edit kind exists
Comparison grid:
| Choice | Current | A | B |
|---|---|---|---|
| Hook shape per layer | one hook per edit kind, registration order | one composite hook per layer | one composite hook per layer |
| Order | undefined | ablate -> add -> read | add -> ablate -> read |
| Meaning | — | Delete removes the trait first; an added knob still applies on top | Delete also cancels any knob on that direction |
| Read measures | — | state after all edits | state after all edits |
| Tests | — | unit: composite hook output equals manual ablate-then-add | unit: manual add-then-ablate |
Question D4:
D4 — E-A1: Inside one layer, in which order should Delete and Update apply?
State: approved
Actual answer: A, "A) Ablate, add, read (recommended)"
Accepted scope: one composite hook per layer, order ablation -> additive edits -> read-out; UI warns when a knob targets a deleted direction; unit test vs manual ablate-then-add

### E-A2: Origin check on mutating localhost routes
Finding: Section 1 finding 2, INFORMATIONAL, 7/10, server.py `/api/stop` (no body, simple cross-site POST); reviewer: Claude
Plan baseline: CEO Section 3 T3 "owner must prove an Origin check on mutating routes" and CEO task T8 (no user answer recorded)
Runtime evidence: server.py has no CORS middleware and no Origin check; JSON POSTs need a preflight and fail cross-site; bodiless /api/stop does not
Comparison grid:
| Choice | Current | A | B |
|---|---|---|---|
| Mutating routes (/api/run, /api/model, /api/stop, later imports) | no Origin check | reject requests whose Origin header is not the server's own origin | unchanged |
| Requests without Origin (curl, tests) | allowed | allowed | allowed |
| Test | — | unit: cross-origin POST -> 403; same-origin -> 200 | none |
Question D5:
D5 — E-A2: Should the server reject mutating requests from other web origins?
State: approved
Actual answer: A, "A) Add Origin check (recommended)"
Accepted scope: middleware rejects mutating requests whose Origin is not the server's own origin; requests without Origin allowed; unit test cross-origin 403, same-origin 200

Dispositions (Section 1): finding 1 accepted (D4); finding 2 accepted (D5); design-twice pick B recorded; typed edits = required work of approved E4/O8.

## Section 2: Code quality

Findings:
1. [INFORMATIONAL] (confidence: 9/10) four activation-capture loops: scripts/derive_llama_recipe.py:100 `def capture_means`, scripts/spike_knob.py `def capture_last` and `def median_norms`, steer.py:321 `def calibrate_layer_medians`. Each registers a hook on every block, runs prompts and removes hooks. Shared-code rubric: 4 verified authored callers; behavior differs in hook point (block input vs output) and reduction (mean over tokens 1:, last token, norm median). Estimated implementation lines: removed ~70, added ~30, saved ~40 [estimated]; tests may grow total change.
2. [INFORMATIONAL] (confidence: 9/10) server.py `except Exception as e:  # noqa: BLE001` in `_run_stream`: carried from CEO Section 2 (owner names classes per capability); no new remedy.

### E-Q1: one activation-capture helper
Finding: Section 2 finding 1, INFORMATIONAL, 9/10; reviewer: Claude
Plan baseline: CEO Section 5 item 2 "One capture helper" (proposal, no user answer)
Runtime evidence: four loops as listed; calibrate_layer_medians is covered by tests/test_calibration.py and feeds the committed calibration cache
Comparison grid:
| Choice | Current | A | B | C |
|---|---|---|---|---|
| Helper | none | Engine.capture(texts, layers, point, reduce) | same helper | none |
| Migrated callers | — | spike, derive, calibration | spike + derive (new/eval code) | none |
| calibrate_layer_medians | own loop | uses helper; cache keys unchanged | unchanged | unchanged |
| Risk | — | touches tested calibration path | low | duplication grows with CRUD |
| Tests | test_calibration.py | helper contract tests + existing calibration tests must pass unchanged | helper contract tests | none |
Question D6:
D6 — E-Q1: Should lobot have one activation-capture helper, and which callers move to it?
State: approved
Actual answer: B, "B) Helper, spike + derive (recommended)"
Accepted scope: Engine.capture(texts, layers, point, reduce); spike and derive migrate; steer.calibrate_layer_medians untouched; helper contract tests

Dispositions (Section 2): finding 1 accepted (D6); finding 2 carried from CEO Section 2 (owner), no new remedy.

## Section 3: Tests

Framework: unittest under tests/ (91 tests today across 9 files); model lane opt-in via MODEL_LOBOTOMY_INTEGRATION=1 (tests/test_steering_loop.py:46). CLAUDE.md has no Testing or Prompt/LLM section.

```
CODE PATHS (waves 0-1)                                   USER FLOWS
[+] engine.py Engine                                      [+] Create a trait
  ├── run(EditSet) composite hook per layer                 ├── [GAP] staged progress 80-90 s, Stop mid-create
  │   ├── [GAP] order ablate->add->read (D4)                ├── [GAP] no verb eligible -> "No usable knob"
  │   ├── [GAP] per-row dose in a batch                      └── [GAP] [→E2E] create -> trait listed -> usable
  │   └── [★★ TESTED] neuron add (test_steering_loop, model) [+] Filmstrip (Update)
  ├── clean() zero hooks of any kind (D2)  [GAP]            ├── [GAP] [→E2E] batched == single ids, every dose (D3)
  ├── capture(...) helper (D6)  [GAP]                        ├── [GAP] mismatch -> sequential fallback (D3)
  └── stream + Stop  [★★★ TESTED] test_server.py              └── [GAP] knob vs prompt verdict shown
[+] traits.py Trait + store                               [+] Guard
  ├── [GAP] per-verb eligibility from report cards (O2)     ├── [GAP] guard on blocks a refusal-weakening edit set
  ├── [GAP] atomic write (temp + rename), no half trait     └── [GAP] research mode allows + labels
  └── [GAP] execution fingerprint mismatch refused (O8)   [+] Security
[+] evaluation.py                                           └── [GAP] cross-origin POST -> 403 (D5)
  ├── [GAP] three splits disjoint; final split untouched (O4)
  ├── [GAP] [→EVAL] judge with question context vs hand labels (O3)
  ├── [GAP] Read gate: behavior-judged, incl. projection-moves-behavior-not (O1)
  └── [GAP] guard check on refusal set for the whole edit set (O5)
[+] app.py loader
  └── [GAP] revision pinned in loaders (O8, app.py:97)

COVERAGE: 3/22 paths tested  |  QUALITY: ★★★:1 ★★:1  |  GAPS: 19 (2 E2E, 1 eval)
Legend: ★★★ behavior + edge + error | ★★ happy path | [→E2E] model lane | [→EVAL] judge eval
```

Required tests carried from approved answers (no new approval needed): D2 registry/clean(); D3 batched == single ids + fallback; D4 composite order; D5 Origin 403/200; D6 helper contract; O1 Read gate negatives; O2 eligibility; O3 judge vs hand labels; O4 split disjointness; O5 guard blocks/allows; O8 fingerprint + revision; S3-1 pickled payload rejected (wave 4).
Regression (existing behavior at risk): tests/test_server.py stream contract and test_steering_loop.py must pass unchanged after the Engine refactor (D2); README reproducibility (D3).
Tests made obsolete: none (test_server.py patches `app` globals; it moves to injecting a fake Engine, same assertions).

Findings:
1. [INFORMATIONAL] (confidence: 8/10) O4 approved three splits but no sizes or pass statistic. The spike's generation check used N_GEN = 4 prompts (scripts/spike_knob.py `N_GEN = 4`); a 4-prompt final split cannot separate a real gain from noise.

### E-T1: final-split size and pass rule
Finding: Section 3 finding 1, INFORMATIONAL, 8/10; reviewer: Claude
Plan baseline: O4 approved "three splits, final untouched; equal tuning budget for prompt baseline" (sizes, statistic unspecified)
Runtime evidence: spike N_GEN = 4; question bank 40 prompts
Comparison grid:
| Choice | Current | A | B |
|---|---|---|---|
| Question bank | 40 | 120 (60 extraction / 30 tuning / 30 final) | 60 (30 / 15 / 15) |
| Pass statistic | judge gain >= 0.30 (mean) | 95% bootstrap CI of judge gain excludes 0 AND mean gain >= 0.30; random control CI includes 0 | mean gain >= 0.30 and control < 0.15 |
| Create time per trait | 82-96 s [measured] | about 3x longer [estimated] | about 1.5x [estimated] |
| Tests | — | statistic unit test with fixed fixtures | threshold unit test |
Question D7:
D7 — E-T1: How large should the final test split be, and what counts as a pass?
State: approved
Actual answer: A, "A) 120 prompts + bootstrap CI (recommended)"
Accepted scope: question bank 120 (60 extraction / 30 tuning / 30 final); pass = 95% bootstrap CI of judge gain excludes 0 AND mean gain >= 0.30; random-control CI includes 0; statistic unit test with fixed fixtures; Create about 3x slower per trait [estimated]

Dispositions (Section 3): finding 1 accepted (D7). Test plan artifact written to ~/.gstack/projects/model-lobotomy/ (eng-review-test-plan).

## Section 4: Performance

Findings:
1. [INFORMATIONAL] (confidence: 7/10) Guard cost: O5 checks refusals for the whole active edit set before each run. Generating replies for a refusal set (for example 20 prompts x 32 tokens at about 34 tok/s on Llama [measured: artifacts/v1-20261029-161530 run.json tok_per_s 34.1]) adds about 20 s per run [estimated] whenever edits change.
2. [INFORMATIONAL] (confidence: 8/10) Create holds the run lock for about 4-5 minutes per trait after D7 [estimated]; runs wait. One model fits MPS at a time, so parallel runs are not possible anyway; the UI must show staged progress and Stop (carried, CEO Section 11).
3. [INFORMATIONAL] (confidence: 6/10, medium confidence, verify) Memory: batch KV cache for 8 rows x 256 tokens on Llama 3B is small next to 6.4 GB weights [estimated]; MPS fragmentation unknown (wave-0 gate 4).

### E-P1: guard check cost
Finding: Section 4 finding 1, INFORMATIONAL, 7/10; reviewer: Claude
Plan baseline: O5 approved "behavioral refusal check on a fixed refusal set for every active intervention set; enforced from wave 1" (method and caching unspecified)
Runtime evidence: Llama generation 34.1 tok/s [measured]
Comparison grid:
| Choice | Current | A | B |
|---|---|---|---|
| Check method | unspecified | first-token refusal probability on the refusal set (prefill only), validated once against full generations + judge | full generation + judge every time |
| Caching | none | cached by edit-set fingerprint (traits, doses, guard version, execution fingerprint) | cached the same way |
| Cost per new edit set | — | about 1-2 s [estimated] | about 20 s [estimated] |
| Validation | — | wave-0: prefill check agrees with full judged check on the refusal set | none needed |
| Tests | — | unit: cache hit/miss on fingerprint; agreement report | unit: cache hit/miss |
Question D8:
D8 — E-P1: How should the refusal guard check an edit set without slowing every run?
State: approved
Actual answer: A, "A) Prefill check + cache (recommended)"
Accepted scope: first-token refusal probability on the fixed refusal set, cached by edit-set fingerprint (traits, doses, guard version, execution fingerprint); wave 0 validates agreement with full generation + judge; falls back to full generation + cache if agreement is poor; unit tests for cache hit/miss and an agreement report

Dispositions (Section 4): finding 1 accepted (D8); finding 2 carried (UI progress + Stop, CEO Section 11); finding 3 pending wave-0 gate 4 measurement.


## Outside voice (eng, Codex gpt-6-astra per probe; completed; verdict: findings P1)

Author verification: scripts/derive_llama_recipe.py:100-115 confirmed (docstring "Mean down_proj-input activation" but `block.register_forward_pre_hook(_cap)` captures the block input, residual width); server.py:172-189 confirmed (stage events around blocking generate; texts arrive at completion). spike_knob.py:114 and :385 citations consistent with the spike.

Factual corrections (no behavior change, recorded without a question):
- OV7: Section 3 diagram overclaimed "stream + Stop ★★★ TESTED". Tested today: stage events and Stop. Token streaming does not exist. The approved live filmstrip (wave 1) therefore requires a per-column token streaming protocol, cancellation and disconnect cleanup, and a test that tokens arrive before generation ends (required work of approved scope).
- OV8: the "Create about 3x slower" estimate (D7 accepted scope) is unsupported. Tuning alone multiplies generation by about 7.5x over the spike; with final evaluation and prompt-baseline tuning, Create is likely 10-20 minutes per trait [estimated; re-measure before gate]. Wave-0 gate adds a full-schedule Create benchmark.

### OV1 (reopens D6 for the derive caller)
Finding: Outside voice 1, High; reviewer: Codex; verified scripts/derive_llama_recipe.py:100-115
Plan baseline: D6 approved "helper; spike and derive migrate"
Runtime evidence: derive captures block input (residual), ranks those channels, then validates them as down_proj neuron IDs
Comparison grid:
| Choice | Current (D6) | A Apply | B Keep | C Investigate | D Defer |
|---|---|---|---|---|---|
| Derive capture point | block input | down_proj input (MLP intermediate), dim asserted = spec.intermediate | block input | measure both on one model first | unresolved |
| Migration to helper | yes | yes, after the fix | yes | after investigation | unresolved |
| Llama preset L11 N1343 | shipped | kept (empirically validated), report notes the old derivation | kept | kept | kept |
| Tests | test_derive.py ranking only | + capture dim assertion test | none | — | — |
Question D9: D9 — OV1: Should the derive script capture real MLP neuron activations before it moves to the shared helper?
State: approved
Actual answer: A, "A) Apply (recommended)"
Accepted scope: derive hooks down_proj input with dim assertion = spec.intermediate, then migrates to Engine.capture; Llama preset kept with a derivation note; capture dim test

### OV2 (reopens D7 control rule)
Finding: Outside voice 2, High; reviewer: Codex
Plan baseline: D7 approved "random-control CI includes 0"
Runtime evidence: n/a (statistical rule)
Comparison grid:
| Choice | Current (D7) | A Apply | B Keep |
|---|---|---|---|
| Control rule | control CI includes 0 | control upper 95% bound < 0.10 (negligible margin) | as D7 |
| Trait vs control | not compared | paired trait-minus-control gain, 95% CI excludes 0 | not compared |
| Trait rule | CI excludes 0 and mean >= 0.30 | unchanged | unchanged |
Question D10: D10 — OV2: Should the random-control rule require proof of no effect instead of accepting uncertainty?
State: approved
Actual answer: A, "A) Apply (recommended)"
Accepted scope: control upper 95% bound < 0.10 AND paired trait-minus-control gain CI excludes 0; trait rule unchanged; statistic unit tests

### OV3 (reopens D8 validation)
Finding: Outside voice 3, High; reviewer: Codex
Plan baseline: D8 approved "wave 0 validates agreement with full generation + judge"
Runtime evidence: n/a
Comparison grid:
| Choice | Current (D8) | A Apply | B Keep |
|---|---|---|---|
| Validation set | unspecified edits | known refusal-weakening edits (incl. llama-refusal preset), unseen combinations and doses, plus harmless edits | unspecified |
| Gate | aggregate agreement | zero missed refusal-weakening edit sets on the validation set; false positives reported | aggregate agreement |
| Fallback | full generation | full generation if any miss | full generation |
Question D11: D11 — OV3: Should the fast guard be gated on missed refusal failures, not overall agreement?
State: approved
Actual answer: A, "A) Apply (recommended)"
Accepted scope: fast guard validated on known refusal-weakening edits (incl. llama-refusal preset), unseen combinations and doses, plus harmless edits; gate zero misses; false positives reported; full-generation fallback on any miss

### OV4 (Create for arbitrary descriptions)
Finding: Outside voice 4, High; reviewer: Codex
Plan baseline: D1 CEO "Create: describe a trait -> contrast prompts -> direction"; O3 judge validated on hand labels (per trait unspecified)
Runtime evidence: spike needs hand-written pos/neg instructions, judge label and probe items per behavior (scripts/spike_knob.py BEHAVIORS)
Comparison grid:
| Choice | Current | A Apply | B Keep | C Investigate | D Defer |
|---|---|---|---|---|---|
| Wave-1 Create input | any description | supported trait families only (validated templates + judge rubric); other text answered "not supported yet" | any description, pipeline undefined | bounded spike first: 2 new traits, model-written instructions, 20 user labels each; Create scope unchanged until answered | unresolved |
| Judge validity for new traits | undefined | per family, hand-validated | undefined | measured in spike | — |
| Headline "type a trait" | yes | yes, within families | yes | decided after spike | — |
Question D12: D12 — OV4: How should Create handle descriptions nobody has validated yet?
State: approved
Actual answer: A, "A) Apply: supported families only"
Accepted scope: wave-1 Create accepts validated trait families only (template + hand-validated judge rubric); other descriptions answered 'not supported yet'; narrows the CEO Create headline by user choice (recommendation was C)

### OV5 (reopens D4 hook sites)
Finding: Outside voice 5, High; reviewer: Codex; steer.py:286 pre-hook on down_proj vs spike block forward hook
Plan baseline: D4 approved "one composite hook per layer, ablation -> additive -> read"
Runtime evidence: neuron edits act on MLP input; vector edits, ablation and read act on block output
Comparison grid:
| Choice | Current (D4) | A Apply | B Keep |
|---|---|---|---|
| Physical sites | one composite hook | two sites per layer, one registry: MLP input (neuron add) and block output (ablate -> vector add -> read) | one composite (not possible without converting neuron edits) |
| Cross-site order | ablate first | neuron add -> ablate -> vector add -> read | as D4 |
| Neuron knob on a deleted direction | still pushes | partly removed by the ablation; UI warns | — |
| Replay of existing neuron recipes | exact | exact (neuron path unchanged) | needs conversion + regression |
| Tests | composite order test | per-site order test; existing steering loop unchanged | — |
Question D13: D13 — OV5: Should lobot keep two hook sites per layer with a fixed cross-site order?
State: approved
Actual answer: A, "A) Apply (recommended)"
Accepted scope: two hook sites per layer under one Engine registry: MLP input (neuron add) -> layer output (ablate -> vector add -> read); UI warns when a neuron knob overlaps a deleted direction; per-site order test; supersedes D4's single composite hook

### OV6 (new policy: final split retirement)
Finding: Outside voice 6, High; reviewer: Codex
Plan baseline: O4 + D7 (splits and sizes); no retirement policy
Runtime evidence: n/a
Comparison grid:
| Choice | Current | A Apply | B Keep |
|---|---|---|---|
| Candidate freeze | none | instructions, layer and dose frozen before the final split is read | none |
| Final evaluations per candidate version | unlimited | one | unlimited |
| After a failed final | retune on same final | new attempt needs fresh confirmation prompts from a reserve pool | retune |
| Starter library report | passing traits | passing traits + number of candidates tried | passing traits |
Question D14: D14 — OV6: Should each final test split be used only once per frozen candidate?
State: approved
Actual answer: A, "A) Apply (recommended)"
Accepted scope: candidates frozen before reading the final split; one final evaluation per frozen version; new attempts use fresh prompts from a reserve pool; starter library reports candidates tried


TODO decision (eng): open-ended Create for unvalidated descriptions -> A) Add to TODOS.md (recommended), user answer 2026-10-29.

Approval readiness (eng): PASS. Checked: E-S1/D1 ("B) Smaller arrangement"), E-SC1/D2 ("A) Engine class owns all"), E-SC3/D3 ("A) Exactness gate + fallback"), E-A1/D4 ("A) Ablate, add, read", superseded in mechanism by D13), E-A2/D5 ("A) Add Origin check"), E-Q1/D6 ("B) Helper, spike + derive"; derive fixed first per D9), E-T1/D7 ("A) 120 prompts + bootstrap CI"; control rule amended by D10), E-P1/D8 ("A) Prefill check + cache"; validation amended by D11), OV1-OV6/D9-D14 ("A" each; D12 "A) Apply: supported families only"). OV7, OV8 factual corrections. No accepted remedy lacks an answer.

## Eng required outputs

### Reality rows (eng)
REALITY: premise-table finding unverified=3 (batched==single on MPS, prefill guard agreement, Create time 10-20 min) scripts/spike_knob.py:385
REALITY: already-done pass no commits for engine.py, traits.py, evaluation.py; local log only (no remote) server.py:1
REALITY: surface-check finding streaming claimed but server emits stage events only; llama-cli not installed server.py:172
REALITY: binding-decisions pass decision log has the CEO record only; eng answers do not overrule it except D12 narrowing Create, recorded as user choice docs/plans/behavior-crud.md:1
REALITY: numbers pass Llama 34.1 tok/s measured; Create 10-20 min estimated, re-measure before gate artifacts/v1-20261029-161530/run.json:1
REALITY: deferred-asks pass no linked issue TODOS.md:1
REALITY: pr-count pass default one PR per wave; waves 0 and 1 are two PRs pyproject.toml:1

### NOT in scope (eng)
- Open-ended Create for unvalidated descriptions: deferred to TODOS.md (D12 + TODO answer).
- Migrating `steer.calibrate_layer_medians` to the capture helper: not chosen (D6 B), it stays as is.
- Converting neuron knobs to residual vectors: not chosen (D13 keeps two sites).
- Baked Delete (E6): deferred by CEO review.

### What already exists (eng)
`steer.register_steering_hooks` (neuron add, steer.py:254) stays the MLP-input site under the Engine registry; `steer.CancelCriteria` and `server.py` stage stream are reused (token streaming is new work, OV7); `steer.calibrate_layer_medians` and its cache stay untouched (D6); spike judge, splits and random control are promoted into evaluation.py with D7/D10/D14 rules; `tests/test_server.py` moves to a fake Engine with the same assertions.

### Diagrams (eng)
```
  EditSet --> Engine.run()
                 |  (registry: every hook kind; Engine.clean() asserts none before calibrate/judge)
                 v
   layer L:  [MLP input: neuron add] --> MLP --> [layer output: ablate -> vector add -> read]
                 |
   batch of doses --(wave-0 gate: ids == single runs?)--> yes: batched | no: sequential (D3)
   guard: edit-set fingerprint --> cache? --> prefill refusal prob on refusal set (D8, validated D11) --> allow | block
   report card: extraction(60) -> tuning(30) -> FREEZE -> final(30, once per frozen version, D14)
                pass = CI(gain)>0 & mean>=0.30 & control upper<0.10 & CI(trait-control)>0 (D7, D10)
```

### Failure modes (eng)
| Path | Realistic failure | Test / handling | User sees |
|---|---|---|---|
| Engine.clean() | a hook kind bypasses registry | registry test + clean() raises | error, run refused |
| Batched filmstrip | fp16 near-tie flips a token | equality gate + model-lane test, sequential fallback | slower filmstrip, exact replay |
| Composite order | neuron knob along deleted direction | per-site order test | UI warning |
| Derive capture | wrong capture point | dim assertion test | derive fails loudly |
| Guard fast check | misses a weakening edit set | zero-miss gate, fallback | "Guard is on: blocked" |
| Report card | peeking via retries | freeze + one final per version + reserve pool | pass/fail with candidates tried |
| Streaming | disconnect mid-stream leaves lock | lock in generator (existing), cleanup test | run ends, lock free |
| Create | 10-20 min, user leaves | staged progress + Stop | progress, can stop |
Critical gaps (no test AND no handling AND silent): 0.

### Worktree parallelization
| Step | Modules touched | Depends on |
|---|---|---|
| Engine refactor (D2, D13, D5) | engine.py, app.py, server.py, tests/ | — |
| Capture helper + derive fix (D6, D9) | engine.py, scripts/ | Engine refactor |
| Evaluation: splits, judge, stats (O3, O4, D7, D10, D14) | evaluation.py, tests/ | — |
| Guard (D8, D11) | evaluation.py | Evaluation |
| Batched filmstrip + streaming (D3, OV7) | engine.py, server.py, web/ | Engine refactor |
Lane A: Engine refactor -> capture helper -> batched filmstrip + streaming (shared engine.py). Lane B: Evaluation -> guard (independent evaluation.py).
Execution order: launch A + B. Merge both. Then wave-0 gates (model runs). Conflict flag: tests/ touched by both lanes; keep new test files separate.

## Implementation Tasks (eng)

- [x] **E1 (P1, human: ~1 day / CC: ~2h)** — engine — Create Engine class owning model, lock and one hook registry with two sites per layer
  - Status 2026-10-30: done on backend-w01 (Engine owns model, tokenizer, lock and the hook registry; EditSet; run, batched, stream, prefill, capture, fingerprint), merged into the frontend branch.
  - Surfaced by: Scope challenge 1 (D2), OV5 (D13)
  - Files: engine.py, app.py, server.py, tests/test_server.py
  - Verify: `uv run python -m unittest discover -s tests -t .`; existing stream and steering-loop tests unchanged
- [x] **E2 (P1, human: ~1h / CC: ~10min)** — server — Add Origin check middleware on mutating routes (eval-gate merge, plus a loopback Host check against DNS rebinding)
  - Surfaced by: Section 1 finding 2 (D5)
  - Files: server.py, tests/test_server.py
  - Verify: cross-origin POST 403, same-origin 200
- [x] **E3 (P1, human: ~2h / CC: ~20min)** — derive — Fix derive capture point to down_proj input, then move spike and derive to Engine.capture
  - Surfaced by: OV1 (D9), Section 2 (D6)
  - Files: scripts/derive_llama_recipe.py, scripts/spike_knob.py, engine.py, tests/test_derive.py
  - Verify: capture dim test; `scripts/derive_llama_recipe.py --self-test`
- [ ] **E4 (P1, human: ~1 day / CC: ~2h)** — evaluation — Build 120-prompt bank, freeze/one-final policy, judge with context, CI rules
  - Surfaced by: Section 3 (D7), OV2 (D10), OV6 (D14), CEO O3/O4
  - Files: evaluation.py, tests/
  - Verify: statistic unit tests with fixed fixtures; judge agreement report on hand labels
- [ ] **E5 (P1, human: ~3h / CC: ~30min)** — guard — Prefill refusal check with fingerprint cache, validated on weakening edits with zero misses
  - Surfaced by: Section 4 (D8), OV3 (D11)
  - Files: evaluation.py, tests/
  - Verify: zero misses on the weakening set; cache hit/miss test
- [ ] **E6 (P1, human: ~3h / CC: ~30min)** — engine — Batched dose hook with exactness gate and sequential fallback
  - Surfaced by: Scope challenge 3 (D3)
  - Files: engine.py, tests/
  - Verify: model-lane test batched ids == single ids for 7 doses
- [ ] **E7 (P1, human: ~1 day / CC: ~2h)** — server/web — Per-column token streaming with cancel and disconnect cleanup
  - Surfaced by: OV7
  - Files: server.py, web/index.html, tests/test_server.py
  - Verify: test that tokens arrive before generation ends; Stop mid-stream frees the lock
- [ ] **E8 (P2, human: ~2h / CC: ~20min)** — wave 0 — Benchmark the full Create schedule on Llama 3B
  - Surfaced by: OV8
  - Files: scripts/
  - Verify: measured minutes per trait recorded in the plan

## Eng completion summary
- Step 0: Scope Challenge — scope accepted as-is (structure B, three modules)
- Reality rows: 7/7 applicable rows emitted, 3 unverified premises
- Architecture Review: 2 issues found
- Code Quality Review: 2 issues found
- Test Review: diagram produced, 19 gaps identified (all mapped to required tests)
- Performance Review: 3 issues found
- NOT in scope: written
- What already exists: written
- TODOS.md updates: 1 item proposed to user (added)
- Failure modes: 0 critical gaps flagged
- Unresolved decisions: 0 in this review
- Outside voice: codex, completed (8 findings; 6 decided A, 2 factual corrections)
- Parallelization: 2 lanes, 2 parallel / 0 sequential
- Lake Score: 10/11 (scored answers D2, D3, D5-D11, D13, D14; D6 chose B at 8/10; D1, D4, D12 unscored)

# Design review (/plan-design-review, 2026-10-29)

Target: this plan, waves 0-1 UI. DESIGN.md (Ward chart) is the calibration source. Mockups: unavailable (no OpenAI key); text-only.
Step 0: 4/10. Focus: all 7 passes (user answer).

## Design outside voices
Codex (completed): APP UI with a persuasive first-run state; no hard rejections; 11 findings (8 High, 3 Medium).
Claude subagent (completed): no hard rejections; 11 findings (3 Critical, 4 High, 4 Medium).
Litmus: brand YES/YES; one anchor NO/YES (disagree); scannable NO/NO; one job NO/NO; cards necessary NO/NO; motion NO/YES (disagree); premium without shadows YES/YES.
Shared factual correction: Section 11 says Create takes 80-90 s; the eng correction OV8 estimates 10-20 min [estimated]. Every Create state below uses 10-20 min until wave 0 measures it.

## Design passes (findings pending individual decisions)
Pass 1 Information architecture: 3/10. Pass 2 States: 2/10. Pass 3 Journey: 4/10. Pass 4 AI slop: 6/10. Pass 5 Design system: 6/10. Pass 6 Responsive/a11y: 3/10.

### Design decisions (each approved individually, 2026-10-29)

| # | Pass | Decision (user answer) |
|---|---|---|
| 1 | IA | 1A Split first: recorded comparison -> pick validated trait -> prompt -> Run comparison -> doses + evidence. Hierarchy: (1) selected dose pink split vs baseline, (2) verdict line, (3) other doses, prompt column, evidence table. Create lives in Library; single-neuron controls under Advanced. |
| 2 | IA | 2A Three readable columns Baseline / Selected dose / Prompt-only + ruled dose strip (per-dose state + split token); expanded filmstrip view with horizontal scroll; phone: selected dose + tabs (Baseline, Prompt) + dose strip. Pink split and the 420 ms cut only on the selected dose; other doses mark their split in ink with a mono token index. |
| 3 | States | 3A Create: estimate shown before start ("about 10-20 min, estimated"); stage rows contrast / capture / tune / freeze / final / report with work units, elapsed mono time, persistent Stop; Run disabled with reason "Create running: stage N of 6"; saved traits browsable; Stop before freeze discards at no cost; Stop during final asks to confirm this version's final check is used up; reload: server continues, page reattaches via a status call. |
| 4 | States | 4A Per column: QUEUED / GENERATING (token count) / COMPLETE / STOPPED at token N (partial kept) / FAILED (cause); new run moves old outputs to a labeled "previous run" state; setting change marks STALE; fallback label "Running doses one at a time (exact replay)"; auto-scroll only while at bottom, else "Follow output". |
| 5 | States | 5A Empty library: recorded comparison stays hero; picker line "No validated trait for this model yet" + one action "Create one from a family"; neuron preset under Advanced. |
| 6 | States | 6A Guard states GUARD ON · NOT CHECKED / CHECKING EDITS (on dose change, ~1-2 s) / CHECK PASSED / BLOCKED (names edit set + measured refusal drop vs threshold; actions Edit doses, Research mode) / CHECK FAILED. Research mode: confirm dialog, session only, masthead "RESEARCH MODE · GUARD OFF", recorded in results, shown in Capture view and clips, never auto-runs. |
| 7 | States | 7A Verdict as mono state key KNOB BETTER / PROMPT BETTER / NO CLEAR WINNER / EVIDENCE UNAVAILABLE + one sentence + effect and side-effect deltas; label "Held-out evaluation · trait vN · dose d", separate from "This prompt"; withheld while streaming or after Stop; STALE on change; prompt column shows its exact instruction. |
| 8 | Journey | 8A Library ruled list: saved traits first (name, model, eligible verbs, version), then "Create new version" per supported family with definition and estimate; unsupported families disabled with reason; optional label inside a family; no free text. |
| 9 | AI slop | 9A Honest trace: flat ink while matching, pink tick at the selected split, solid "diverged" band after it (no jitter); ink ticks for other doses' splits with signed dose on hover/focus. |
| 10 | Design system | 10A Evidence table (no cards): rows Read / Increase / Decrease / Delete; columns state key (PASS / FAIL / NOT TESTED / LATER WAVE), effect + 95% CI, control upper bound, side effects, tested dose range; footer fingerprint, trait version, candidates tried; failed/untested verbs disable their control with reason inline; "No usable knob" reports viewable. |
| 11 | Design system | 11A Trait palette in DESIGN.md before wave 1: `trait-1`..`trait-6` muted ink-family hues, light + dark, >= 3:1 vs sheet, always with a text label, never pink / oxblood / state hues. |
| 12 | Design system | 12A Self-host fonts: woff2 for used weights in web/fonts, @font-face, remove Google Fonts link, verify first screen offline. |
| 13 | A11y | 13A Columns not live while streaming; one polite region for stage changes and completion; dose strip = keyboard radiogroup, 44px targets, labels with signed dose + state; Create stages as ordered list with aria-current; focus moves to verdict after a run; reduced motion shows the cut's final state. |
| 14 | Decisions | 14A Wave-2 token inspector contract: opens on hover, focus or tap; pin + Esc; names tokens for multi-token words; "Observed projection" (with scale) separate from "Applied dose"; missing = n/a. Wave-1 completion does not depend on it. |

Scope correction (no question): dial hover values belong to wave 2 per the approved wave table; this review's focus line listed them under wave 1 in error.

### Wave-1 screen (1440 px)

```
 lobot                       PATIENT: Llama-3.2-3B ▾   Library   History   [RESEARCH MODE · GUARD OFF]*
 ───────────────────────────────────────────────────────────────────────────────────────────────────────
 Trait: Paranoid v2 ▾  [Increase · Decrease · Read=LATER WAVE]   Prompt [.................]   (Run comparison)
 Dose strip: −1.0 │ −0.5 │ 0 │ +0.25 │ +0.5● │ +0.75 │ +1.0    (state + split token per dose; ● = selected)
 State bar: COMPLETE · VERDICT: PROMPT BETTER · Held-out evaluation · trait v2 · dose +0.5   GUARD: CHECK PASSED
 Trace ──────────────────▌pink tick │ ink ticks ▬▬▬▬▬▬▬▬▬▬▬▬ diverged ▬▬▬▬▬▬▬▬
 ┌ BASELINE ────────────┬ DOSE +0.50 (selected) ─────┬ PROMPT-ONLY "Answer as someone paranoid…" ┐
 │ ...                  │ ... [pink split] ...       │ ...                                         │
 └──────────────────────┴────────────────────────────┴─────────────────────────────────────────────┘
 Evidence  Read │ Increase │ Decrease │ Delete   (state key, effect + CI, control bound, side effects, doses)
 * only when research mode is on
```
Phone (< 720 px): trait + prompt + Run, dose strip (scrolls), state bar, then the selected dose; tabs switch to Baseline or Prompt-only; evidence table becomes stacked label/value rows.

### Journey storyboard (wave 1)

| Step | User does | User feels | Plan specifies |
|---|---|---|---|
| 1 | Opens lobot | Curious, "what is this?" | Recorded comparison as hero (1A), honest trace (9A) |
| 2 | Picks a validated trait | Oriented | Trait picker; empty state (5A) |
| 3 | Types a prompt, runs | Anticipation | Per-column streaming states (4A), guard check (6A) |
| 4 | Watches the selected dose split | "Whoa" | Pink split + cut on selected only (2A) |
| 5 | Reads the verdict | Trust | Scoped verdict (7A), evidence table (10A) |
| 6 | Tries other doses | Control | Dose strip, keyboard radiogroup (2A, 13A) |
| 7 | Creates a new version (later) | Patience, not doubt | Library list (8A), Create stages + honest Stop (3A) |
5-second: the split. 5-minute: dose control and verdict. Long-term: trustworthy evidence and replay.

### Interaction states (wave 1)

| Feature | Loading | Empty | Error | Success | Partial |
|---|---|---|---|---|---|
| Comparison columns | QUEUED / GENERATING n tokens | recorded example | FAILED (cause) | COMPLETE + split | STOPPED at token N, STALE |
| Verdict | withheld while streaming | EVIDENCE UNAVAILABLE | EVIDENCE UNAVAILABLE (reason) | KNOB / PROMPT BETTER / NO CLEAR WINNER | withheld after Stop |
| Guard | CHECKING EDITS | GUARD ON · NOT CHECKED | CHECK FAILED | CHECK PASSED | BLOCKED (named edits) |
| Create | stage rows + elapsed | Library list | stage FAILED (cause), no trait written | trait + evidence table | Stopped before freeze (discarded) |
| Library | — | "No validated trait for this model yet" + Create | — | saved traits | unsupported families disabled |

## Design review: NOT in scope
- Wave-2 dials and chat UI beyond the token-inspector contract (14A).
- Mockups: not generated (no OpenAI key); run `/design-shotgun` after `$D setup` for visual exploration.
- Open-ended Create UI (deferred with the TODO from the eng review).

## Design review: what already exists
DESIGN.md tokens (ground, sheet, ink, primary, cut, state tints), the run-state bar and step list, the capture view, the split and cut rendering, the Pretext layout, theme toggle, recipe/export rows in `web/index.html`. Reuse all; new parts use the same state-key and ruled-table vocabulary.

## Implementation Tasks (design)

- [ ] **U1 (P1, human: ~1 day / CC: ~2h)** — web — Wave-1 layout: hierarchy, 3 columns + dose strip, expanded filmstrip, phone tabs
  - Surfaced by: Pass 1 (decisions 1, 2)
  - Files: web/index.html
  - Verify: screenshots at 375 / 768 / 1440; pink only on selected dose
- [ ] **U2 (P1, human: ~1 day / CC: ~2h)** — web — Column, verdict, guard and empty states
  - Surfaced by: Pass 2 (decisions 4, 5, 6, 7)
  - Files: web/index.html, server.py
  - Verify: each state reachable in no-model mode or with a stubbed stream
- [ ] **U3 (P1, human: ~1 day / CC: ~2h)** — web/server — Library list and Create progress with honest Stop and reattach
  - Surfaced by: Pass 2/3 (decisions 3, 8)
  - Files: web/index.html, server.py
  - Verify: reload mid-Create reattaches; Stop confirm only during final stage
- [ ] **U4 (P1, human: ~2h / CC: ~20min)** — web — Honest agreement trace
  - Surfaced by: Pass 4 (decision 9)
  - Files: web/index.html
  - Verify: no random jitter in drawTrace; ticks match split tokens
- [ ] **U5 (P1, human: ~3h / CC: ~30min)** — web — Evidence table per verb
  - Surfaced by: Pass 5 (decision 10)
  - Files: web/index.html
  - Verify: failed verb disables its control with reason
- [ ] **U6 (P1, human: ~2h / CC: ~20min)** — design system — Trait palette tokens in DESIGN.md and CSS
  - Surfaced by: Pass 5 (decision 11)
  - Files: DESIGN.md, web/index.html
  - Verify: contrast >= 3:1 vs sheet in both themes; design lint
- [ ] **U7 (P2, human: ~1h / CC: ~15min)** — web — Self-host fonts
  - Surfaced by: Pass 5 (decision 12)
  - Files: web/fonts/, web/index.html
  - Verify: first screen correct with network off
- [ ] **U8 (P1, human: ~3h / CC: ~30min)** — web — Accessibility for streaming, dose strip and Create
  - Surfaced by: Pass 6 (decision 13)
  - Files: web/index.html
  - Verify: keyboard-only run; screen reader hears stage announcements only

## Design Review Completion Summary

```
  +====================================================================+
  |         DESIGN PLAN REVIEW — COMPLETION SUMMARY                    |
  +====================================================================+
  | System Audit         | DESIGN.md present; wave 0-1 UI scope          |
  | Step 0               | 4/10; all 7 passes                            |
  | Pass 1  (Info Arch)  | 3/10 → 8/10 after fixes                       |
  | Pass 2  (States)     | 2/10 → 9/10 after fixes                       |
  | Pass 3  (Journey)    | 4/10 → 9/10 after fixes                       |
  | Pass 4  (AI Slop)    | 6/10 → 9/10 after fixes                       |
  | Pass 5  (Design Sys) | 6/10 → 9/10 after fixes                       |
  | Pass 6  (Responsive) | 3/10 → 9/10 after fixes                       |
  | Pass 7  (Decisions)  | 1 resolved, 0 deferred                        |
  +--------------------------------------------------------------------+
  | NOT in scope         | written (3 items)                             |
  | What already exists  | written                                       |
  | TODOS.md updates     | 0 items proposed                              |
  | Approved Mockups     | 0 generated (no OpenAI key), 0 approved       |
  | Decisions made       | 14 added to plan                              |
  | Decisions deferred   | 0                                             |
  | Overall design score | 2/10 → 8/10                                   |
  +====================================================================+
```
Remaining below 10: Pass 1 at 8 (section headings for scanning are specified only in the wireframe, not as copy); visual quality is unverified without mockups.

Unresolved decisions (design): none.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 | ISSUES OPEN | 8 proposals, 7 accepted, 1 deferred |
| Outside Review | codex (plan-review, automatic, CEO + Eng) | Independent 2nd opinion | 2 | issues_found | CEO: 8 findings; 7 resolved; 1 unresolved. Eng: 8 findings; 8 resolved; 0 unresolved |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | ISSUES OPEN | 26 issues, 0 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 1 | CLEAR | score: 2/10 → 8/10, 14 decisions |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | not run |

- **OUTSIDE COVERAGE:** codex plan-review completed for CEO and Eng phases; design phase: codex completed (11 findings) and native Claude subagent completed (11 findings), all folded into 14 approved design decisions.
- **CROSS-MODEL:** in the design phase both voices agreed on the stale Create timing, the filmstrip layout, the evidence table over cards, guard states and wave placement of hover; they disagreed on whether the current anchor and motion already work (Codex yes, Claude no); the selected-dose rule (2A, 9A) resolves both.
- **VERDICT:** DESIGN CLEARED; eng review required (eng review ISSUES OPEN means mapped work, not open choices).

**UNRESOLVED DECISIONS:**
- + 1 unresolved from prior reviews
