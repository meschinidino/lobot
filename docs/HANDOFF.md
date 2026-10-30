# Handoff: lobot (2026-10-30)

## Goal

See a behavior inside an open model, change it, prove what the change does,
keep it safe, and share it as a patch others can re-check. The 10X demo is
the loop Read → Steer → Guard → Prove on Llama 3.2 3B, as a recorded 3-minute
video. Later stages (100X, 1000X) lead to a signed, re-verifiable registry of
behavior patches.

## Read first

1. `docs/roadmap/README.md` — stages, decisions, lanes, schedule, rules for agents.
2. `docs/roadmap/contracts.md` — shared routes, signatures and file formats.
3. `docs/roadmap/specs/` — one spec per workstream (R1-R3, D1, E1, S1, P1, J1, U1-U3, V1).
4. `DESIGN.md` — "the fork" workbench. All UI follows it.
5. `docs/api.md` — the routes as built.
6. `docs/plans/lobot-10x-plan.md` — the reviewed plan; decisions D1-D14 still bind.

## State

- Integration branch: `claude/admiring-mendel-mn88n8`. Lane branches
  (`lane/<ID>-<name>`) start from it and merge back through Claude.
- Built: the Engine (edits, capture, `clean()`), Create with report cards,
  Update (multi-dose filmstrip), refusal guard (opt-in), `.lobot` patches and
  llama.cpp export, read dials (`dial_trait`), click-a-word on the prompt
  (`POST /api/neurons`), chat, replay mode, the Origin check.
- UI: `web/index.html` is wired to `/api/run` (sentence, knob, experiment
  tree, guard states, share card). Create, Update, evidence, reader and dials
  in the UI are lanes U1-U3 (Claude).
- Recordings: `web/sweep.json` (Qwen) and `web/sweep.llama-32-3b-instruct.json` (Llama).
- Open question O6 (chat cache semantics): `docs/open-questions.md`.

## Rules

- Simplified Technical English: short sentences, active voice.
- Never claim "verified" for simulated or replayed results; say what was simulated.
- No competition or event references in repo files.
- Commits: author is the repo's configured user. No co-author or tool-attribution lines.
- Never force-push or rebase a shared branch.
- One model in memory at a time: book `docs/roadmap/model-lane.md` before loading weights.
