# Handoff: lobot behavior-CRUD work (2026-10-29)

## Read first
1. `docs/plans/lobot-10x-plan.md` — the full plan with CEO, eng and design reviews and every decision (D1-D14 eng, design decisions 1-14).
2. `docs/designs/behavior-crud.md` — the short scope summary.
3. `DESIGN.md` — the "Ward chart" design system. All UI follows it.
4. `TODOS.md` — deferred work.

## State
- `main` (latest `334fbd0`): web UI (`server.py`, `web/index.html`), Stop, DESIGN.md, plan docs, `scripts/spike_knob.py`.
- `eval-gate` (`971a529`, by another agent): `evaluation.py` (splits, bootstrap CI, report-card rule, final-split ledger) + same-origin POST middleware. Not reviewed or merged yet. Review it against eng decisions D5, D7, D10, D14 before merging.
- Nothing pushed by Claude. Model weights and `logs/` are local only.

## Next step (design)
Build `web/prototype.html`: a static, clickable prototype of the wave-1 main screen with recorded data (`web/example.json`), no backend:
- Baseline | Selected dose | Prompt-only columns; ruled dose strip with per-dose state and split token (design 2A)
- pink split and cut only on the selected dose; other splits in ink (2A)
- verdict state key in the run-state bar, scoped "Held-out evaluation · trait vN · dose d" (7A)
- per-verb evidence table: Read / Increase / Decrease / Delete (10A)
- honest agreement trace: flat, pink tick, solid diverged band, no jitter (9A)
Then, in a second pass: guard states (6A), Create stage rows (3A), Library list (8A), empty state (5A), phone layout (2A), accessibility (13A).
Verify with screenshots at 375 / 768 / 1440 px and the design lint (`gstack-design-lint web/prototype.html --render`).

## After the prototype
Wave 0 lane A: E3 derive capture-point fix (D9), then E1 Engine refactor (D2, D13) after `eval-gate` is merged (both touch `server.py`).

## Rules
- Writing style: Simplified Technical English (short sentences, active voice).
- Never claim "verified" for simulated results; say what was simulated.
- Commit only when asked. No competition or event references in repo files.
