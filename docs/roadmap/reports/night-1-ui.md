# UI and integration report (2026-10-30)

Owner: Claude (UI lead). Branch: `claude/admiring-mendel-mn88n8`.
Read this first if you continue the UI work. All UI screenshots so far came
from the stub server (fake data). Nothing in the UI is checked on real weights yet.

## Done and pushed

| Lane | What | Commit |
|---|---|---|
| U1 | Behaviors on the shelf, New behavior page, Create progress (6 stages, Stop, reattach), behavior evidence page, behavior edits in the sentence, "Run at 5 doses" (one `/api/update` batch, streams into the tree), guard key next to Run, research mode dialog and banner | `4664c10` |
| U2 | Word inspector lists the top neurons at a clicked word (`/api/neurons` with `reply` + `reply_char`), neuron card (pushes toward, fires on), Push, "Find neurons that push…" on the shelf, lift proof line after a run | `b1a5fbd` |
| U3 | "+ read" in the sentence; channel strip (band, pen line, onset tick) under each run's words from `dial_trait`; inspector shows the value at a word | `cfa6658` |
| Server | `_latest_knob` only returns a knob for the active model; `/api/neurons` accepts `reply_char` and refuses in replay mode | `0ab7684`, `636227b` |
| Merges | R1 (reader core), R2 (reader API, merged by Muse) | |

Checks run for every UI change: design lint 0 blocks / 0 asks
(`gstack-design-lint web/index.html --render`), Playwright flows with no page
errors, no horizontal scroll at 375 / 768 / 1440 px, full unit suite passes
(289 tests, 5 skips).

Later the same night: share card shows the dose (one edit) and the measured
lift proof, behavior runs get a post text, Run works with no edit (run 1 only,
the video's first frame), and `docs/roadmap/demo-script.md` gives the video
click by click.

## Contract changes made tonight (backend must follow)

1. `POST /api/neurons` takes `reply_char` instead of `token_index` (done in server).
2. D1: `complete.result.dial.baseline` and `.steered` are lists of
   `{"text", "value"}`, one per generated token. **Not built yet.** Until it is,
   the steered strip uses the streamed `token` events (which already carry
   `dial`), and run 1 (no edit) shows no strip.

## Left to do, in order

1. **Real-model pass (needs the Mac, Llama loaded).** Start `uv run python server.py`
   with Llama active and check, in the browser:
   - click a word in an answer → neuron list appears (R2 + `reply_char` on real tokens);
   - open a card → "Fires on" needs the R3 cache; without it the card says so;
   - push a neuron from the list, Run → lift proof line appears (`/api/lift`);
   - "+ read" with a real Create behavior → strip under the steered run;
   - one guard block, one 5-dose behavior run, one Create started and stopped.
   Fix what breaks. The likely spots: `reply_char` mapping on Llama's tokenizer
   (`server._reply_token_at`), and `plainTok` in `web/index.html` (cleans token
   prefixes like `Ġ` in "pushes toward" lists).
2. **D1 baseline values** (backend, Muse): then run 1 gets its strip too. No UI change needed.
3. **E1 evidence API** (backend): the evidence page and "Download patch" then show
   real data. The UI already calls `/api/traits`, `/api/traits/{id}/report`,
   `/api/traits/{id}/patch` and falls back to jobs when they 404.
4. **V1 video** (`docs/roadmap/specs/V1-demo.md`): needs the R3 demo neuron and
   recordings from `scripts/record_sweep.py`. Then: collapse the shelf (recording
   view), check the share card with the real run, write the X post text, record.
   Replay mode does not serve `/api/neurons` or `/api/lift` (they need the model);
   neuron cards replay from `store/reader/<slug>/cards.json`. So record the "Read"
   beat live, or have R3 write cards.json for the demo neurons.
5. Small polish if time allows: the 375 px evidence table scrolls sideways inside its box.

## How to work on the UI without a model

```bash
uv run python scripts/ui_stub.py            # fake API + web/ on http://127.0.0.1:7861
STUB_NO_E1=1 PORT=7863 uv run python scripts/ui_stub.py   # as if the evidence API is missing
```

Flow scripts (Playwright; run from a scratch folder, they write PNGs there):

```bash
mkdir -p u1 u2 && node <repo>/scripts/ui_flows/u1.js u1   # U1 flows
node <repo>/scripts/ui_flows/u2.js                         # U2 reader flows -> u2/
node <repo>/scripts/ui_flows/u3.js                         # U3 dials -> u2/
```

The stub keeps state in memory; restart it for a clean run. Its numbers are
fake. Never show them as results.

## Rules that still hold

DESIGN.md for every UI change (one pink event per screen: the split; no
dashboard; plain words). Simplified Technical English. No attribution lines in
commits. Never force-push. No event references in repo files.
