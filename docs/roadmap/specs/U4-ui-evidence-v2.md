# U4 UI: evidence page v2 (prompt comparison, fluency, all screens, repair)

Owner: Muse (coding agent). Lane: UI and no-model (replay). One model
slot only to record replays. Depends on: PP, FD, GR. Time: two days.

Read [docs/DESIGN.md](../../DESIGN.md) first. Ask the user before you
leave it.

## Problem

The backend now measures more than the UI shows:

1. The evidence page has a "Side effects" column that says "not
   measured yet". PP measures relevance and fluency per dose.
2. The UI never shows the prompt comparison (PP). That comparison is
   the answer to "why not use a prompt?".
3. The UI does not show the dose rule or the fluency at the chosen
   dose (FD). An old patch at a broken dose looks the same as a new one.
4. The guard row shows the refusal screen only. Reports also hold
   honesty, privacy (with control flips) and harmful screens.
5. The prompt baseline row shows the drafted instruction even when it
   is not an instruction ("Howdy ma'am ..."). PP added `prompt:label`.
6. A flagged patch has no repair action. GR exists only as a script.
7. Create from the server never runs the Pareto stage.

## Backend changes

1. `CreateRequest.pareto: bool | None = None`. `None` means on for the
   `full` schedule and off for `fast`. Pass it to `run_trait`.
2. Add `"pareto"` and `"guard"` to `create_job.STAGES` and to the UI
   stage list, with progress counts.
3. `POST /api/traits/{job_id}/pareto`: runs `pareto.run_pareto` for an
   old job as a background job (same runner rules as Create: one model
   job, stop, reattach). It writes `pareto.json` and the report
   `pareto` block.
4. `POST /api/traits/{job_id}/repair` with `{"method": "gradient"}`:
   runs the `scripts/repair_patch.py` logic as a background job. Move
   the script body into `repair.run_repair_job(...)` so the route and
   the script share it. `GET /api/traits/{job_id}/repair` returns the
   record (`repair_<method>.json`) and, when repaired, the patch link.
5. `GET /api/traits/{job_id}/report` includes `pareto`, `tuning.rule`,
   the chosen tuning cell (`tuning.chosen = {layer, dose, judge, nll,
   fluency}`) and every guard screen.
6. Document every new or changed route in `docs/reference/api.md`.

## Evidence page changes

1. **Side effects column.** For Increase, show relevance and fluency at
   the chosen dose from the Pareto point `vector@<dose>`, with the base
   value. Example: "fluency 0.95 (base 0.96), relevance 0.90 (0.93)".
   Without a Pareto block, keep "not measured" and show the button
   "Compare with a prompt".
2. **Compared with a prompt.** A new section under the table:
   - A small SVG chart. X axis: side-effect cost (0 to 1). Y axis:
     effect (trait gain). Vector doses join in one line (pen-2), each
     point labeled with its dose. Prompt points use pen-4. The chosen
     dose has a ring in the cut color. Pareto-front points are filled;
     the others are hollow.
   - One sentence for the verdict, in plain words:
     `prompt_dominates`: "A plain prompt gives more effect with fewer
     side effects at every dose."
     `vector_dominates`: "The patch gives more effect with fewer side
     effects than a plain prompt."
     `tradeoff`: "Neither wins on both: the prompt gives more effect,
     the patch at dose X costs less." (fill from the numbers).
   - A table with the same rows, for screen readers and for exact
     values. The chart is `aria-hidden`; the table is the source.
3. **Dose rule line** under the meta line: "Dose chosen by the
   fluency-aware rule: fluency 0.95 at the chosen dose." For a report
   without `tuning.rule`, show a warning state: "Chosen before the
   fluency check. Re-run Create or compare with a prompt to see the
   text quality."
4. **Guard block.** One row per screen: name, state (PASSED, WOULD
   BLOCK, BLOCKED), the measure (refusal kept at N%, honesty flips N
   of 8, privacy leaks N of 40 with controls [a, b, c]), and whether
   full generation confirmed it.
5. **Prompt baseline.** Show both instructions: the drafted one and
   the label one, each with its effect and CI. When the drafted one
   reads as a sample reply (not an instruction), add the meta note
   "this baseline is a sample line, not an instruction".
6. **Repair.** When any screen flags, show "Try repair (about N min)".
   During the job, show progress. After it, show the candidate table
   (method, dose factor, flips per screen, tuning gain) and the
   status. When `repaired`, show "Download repaired patch".
7. **Create progress** lists the two new stages.

## Replay

Record the evidence for `logs/fd_pirate_llama` (FD pirate, Pareto
included), `logs/c3_cowboy_llama` (pre-FD, privacy flag) and the GR
pirate repair record. With `MODEL_LOBOTOMY_REPLAY=1`, the page must
render all of it without weights. Mark replayed data as recorded in
the UI, as the other replay screens do.

## Tests

- Server: route tests with fake runners for `/pareto` and `/repair`
  (start, status, stop, unknown job, job without a patch).
- Report shape: a test that `report` returns `pareto`, `tuning.rule`,
  `tuning.chosen` and all screens.
- UI: a pure function `paretoSentence(p)` and `chartPoints(p, w, h)`,
  tested with a small JS test runner or a Python test that runs the
  page script through the existing harness. If none exists, add a
  `node --test` file under `web/tests/`.

## Acceptance

1. Full suite passes; the new routes and functions have tests.
2. In the browser (replay mode), the pirate FD page shows the chart,
   the verdict sentence and a table whose numbers equal `report.json`
   to 3 decimals. Check at 375 px and at desktop width, light and
   dark. No horizontal page scroll at 375 px.
3. The cowboy page shows privacy as a separate guard row with its
   control counts.
4. The repair section shows the recorded pirate GR candidates and
   status `unrepaired` with the numbers from `reports/GR.md`.
5. A screenshot of each state goes in `reports/U4.md`.

## Known risk

The chart is new to the design. If it does not fit the "printed flat"
look, keep the table and drop the chart. Ask the user first.
