# U1 UI: wave-1 wiring (Create, Update, guard, evidence)

Owner: Claude (UI lead). Lane: UI. Depends on: contracts; existing wave-1 API
(`docs/api.md`). Time: one and a half days.

## Goal

Turn the prototype's designed states into working screens on the real API:
Create a behavior with live progress, run an Update filmstrip into the
experiment tree, switch the guard and research mode, and read a trait's
evidence card.

## Taste rules (from DESIGN.md and the 2026-10-30 review)

- Not a dashboard: the main area holds one object (the experiment tree), the
  words, one knob, one number. No tiles, meter rows or number sidebars.
- Dark by default. One pink event per screen: the selected run's split.
- Plain words: MODEL, behavior, Dark/Light. No patient, ward, vitals, theatre.
- Every illustrative value is labelled; every recorded or measured value says
  so. Never show a number the backend did not produce.
- The sentence is the only place edits are set; the shelf manages models and
  behaviors.

## Deliverables (web/index.html, plus web/prototype.src.html where the design changes)

1. **Shelf behaviors** from `GET /api/traits?model_id=` (E1). Row: name,
   version, one state key from the cards (PASS, DECREASE FAIL, READ LATER,
   GUARD). Until E1 lands, `GET /api/jobs` with `status: done`.
2. **New behavior** from `GET /api/families` (validated only; others shown
   disabled with "not supported yet"). Create posts `/api/create`, then polls
   `GET /api/create/{id}` every 2 s: six stage rows mapped from the backend
   stages (extract, tune, read_gate, final, baseline, verdict), progress
   `done/total`, elapsed, Stop (`POST /api/create/{id}/stop`; confirm during
   final). Reattach on load from `GET /api/jobs` (any job not done).
   Estimate shown before start: "about 50 min on Llama 3.2 3B, measured".
3. **Use a behavior** in the sentence ("+ edit" lists saved behaviors): an edit
   of kind behavior with a dose. Running it calls `/api/update` with
   `trait` and one dose, and the result becomes a branch in the tree (same
   shape as `/api/run` runs; baseline from the experiment's run 1 or a
   baseline-dose column).
4. **Filmstrip**: "Run at 5 doses" sends `/api/update` with 5 doses; tokens
   stream into 5 planned branches at once; each completes into a branch.
5. **Guard switch**: state key next to Run (6A states); the request sends
   `guard` (default from `/api/state.guard_default`) and `research_mode`.
   Research mode: confirm dialog, session only, banner, recorded per run.
   A blocked event shows the safe `display_prompt`, the measured ratio and the
   limit.
6. **Evidence card** page for a behavior from `GET /api/traits/{id}/report`:
   the four verbs as rows (state key, effect with 95% CI, control bound, side
   effects, tested doses), the guard result, the judge, the fingerprint, and
   "Download patch" (`/api/traits/{id}/patch`).

## Checks (each must pass before merge)

- Stub server (scratch) and replay mode: every flow above, scripted with
  Playwright: no page errors, no horizontal scroll at 375, 768, 1440 px.
- `gstack-design-lint web/index.html --render`: 0 blocks, 0 asks.
- Keyboard only: Create, Stop, Run, the guard switch, the dialog (Esc closes).
- Real API (model slot shared with V1): one Create on Llama started and
  stopped; one filmstrip; one guard block.
- Screenshots of every state sent to the user; DESIGN.md updated for any new
  component.

## Out of scope

Reader (U2), dials (U3), chat (later.md).
