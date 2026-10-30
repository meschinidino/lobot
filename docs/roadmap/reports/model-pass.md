# Model pass: UI smoke test (live + replay, Llama 3.2 3B)

Date: 2026-10-30. Server: `uv run python server.py` on this
machine, then `MODEL_LOBOTOMY_REPLAY=1` for the replay pass.
Model: meta-llama/Llama-3.2-3B-Instruct. Driver: Playwright
scripts in /tmp (not committed). Screenshots in /tmp/smoke-live
and /tmp/smoke-replay (not committed). I did not edit `web/`.

## Live: what worked

- Boot, model list, switch to Llama, prompt run. Run 1
  (no edit) returns the baseline weekend answer.
- Find "storm" ranks L19N4330 first ("storm" is its #5 word).
- The neuron card opens with pushes_toward and fires_on.
  Push adds L19N4330 at +1.00 to the sentence.
- Research-mode run at 1.0 forks ("I'm a stormy weather
  system..."). The proof line appears after ~15 s: "The
  reader predicted "storm" for L19N4330; pushing it at +1.00
  made "storm" 17550x more likely as the next word (8 neutral
  prompts, measured)."
- Clicking "stormy" opens the inspector with a loading state.
- Evidence page renders the real pirate report: 4 rows,
  judge agreement 1.00, create time 48:45, Download patch
  link, guard row, fingerprint row.

## Live: breaks

B1 (blocks the scripted demo). Steer at 1.0 is BLOCKED with
the guard on. Steps: push L19N4330, knob 1.0, Run, guard on.
Server: 200 with `blocked: true` (verified full-confirm
flipped a borderline prompt; prefill alone passes at
min_ratio 0.787). No server error. Consequence: beats 2 and
3b ("back to 1.0, CHECK PASSED") cannot run guarded. The
verified check also blocks 0.5 and 0.25 for this neuron
(R3.md). Workarounds: research mode for the steer beat, or
an S1 follow-up on borderline flips, or guard beats on
L11N1343 (0.0 passes verified, 2.0 blocks verified).

B2 (misleading note). The BLOCKED note quotes passing
prefill numbers: "the model's chance of refusing fell to
77% of its usual level ... the limit is 50%." The block came
from the full-generation confirm, not the quoted ratio. The
note should name the flipped prompt and the confirm step.

B3 (minor, inference). Run stays enabled while a
click-inspect request holds the engine; pressing Run then
returns 409 ("A run is in progress") and the status shows
ERROR. Steps: click a word, press Run within ~10 s. The UI
should disable Run until the inspector resolves.

B4 (wording, backend + UI). Evidence shows "Refusal guard
PASSED, refusal kept at 19% of its usual level". 19% is
under the 50% limit; the job ran unenforced
(`would_block: true`). An unenforced would-block should not
render as PASSED.

B5 (UI render). Evidence shows "Model fingerprint: [object
Object]". The object needs stringifying.

B6 (harmless). One boot 404: no favicon link, the browser
requests /favicon.ico.

## Replay: what worked

- Boot shows REPLAY. The recorded tree renders (baseline +
  5 storm runs from the committed sweep).
- Click-a-word refuses cleanly: "Replay mode loads no
  weights, and the neurons at a word need the model."
- Find refuses cleanly: "Find needs the model. Replay mode
  has no recordings for it."
- Evidence page serves the pirate report from disk.
- `GET /api/neuron/19/4330` returns the recorded card
  (200, up " storm", fires " storm").

## Replay: breaks

R1 (blocks the recorded demo). Neuron cards are unreachable
in the replay UI. The only card links live in the inspector
list and the find results, and both refuse in replay mode.
`cards.json` is served but has no UI path. The recorded
demo needs one (for example, link run headers to the card).

## Evidence-quality note (not a UI bug)

The pirate evidence rows are mostly red: Increase FAIL
("the random-direction control moved it too", +0.77
[+0.73, +0.81]), Decrease FAIL (-0.10), Delete FAIL (-0.01).
Only Read passes. The demo's Prove beat ("point at the
green rows") has no green rows with this trait. Either a
different family needs a real Create (~50 min each) or the
beat changes.

## Day-2 re-run (2026-10-30, after 809bf4a)

Same method (Playwright scripts `/tmp/day2-*.js`, shots in
`/tmp/smoke-day2-live` and `/tmp/smoke-day2-replay`). Live
server then replay server, Llama 3.2 3B. I did not edit
`web/`.

Fixes confirmed:

- R1 fixed. Replay: click a word, the inspector shows
  "Applied edits in this run L19N4330 +2.00" as a link, and
  the link opens the full card from `cards.json` ("Pushes
  toward storm, storms, hurricane..."; "Fires on...").
- B2 fixed. The storm 1.0 BLOCKED note now reads: "The
  refusal guard blocked L19N4330 +1.00. The first-token
  check passed (refusal at 77% of its usual level; the
  limit is 50%), but in the full-answer check a test
  prompt no longer got a refusal. Lower the dose, or use
  Research mode." It names the confirm step.
- B3 fixed. Clicking a word disables Run with "Run waits
  while the model reads the neurons at a word."
- B4 fixed. Evidence shows "Refusal guard: WOULD BLOCK,
  not enforced in this Create; refusal kept at 19% of its
  usual level, first-token check only".
- B5 fixed. Evidence shows the fingerprint stringified
  ("model_id: meta-llama/Llama-3.2-3B-Instruct; revision:
  0cb88a4f...; template_sha256: ...; thinking: false;
  dtype: torch.float16; quant: null; device: mps:0;
  cal-code: t1-v2").
- B1 stands, now explained: storm 1.0 still blocks the
  verified guard, and S1-followup.md shows it is real
  refusal loss (keylogger complied at 0.5), not a
  borderline flip. The scripted steer beat needs research
  mode; the guard beats use L11N1343.
- B6 persists (two favicon 404s per boot, harmless).

Also verified live: research-mode fork at 1.0 ("I'm a
stormy weather system...") with the lift proof line after
~20 s ("...made "storm" 17550x more likely...", same
17550x as last night, deterministic). The new sweeps seed
the tree on first visit (baseline + 1.0 + 2.0). Guard
beats work server-side: `POST /api/run` L11N1343 at 2.0
→ BLOCKED (prefill, min_ratio 0.004, 7 prompts); at 0.0
→ COMPLETE (verified True, min_ratio 1.0, no split).

New findings (day 2):

D1 (blocks the scripted guard beat). "+ edit" then "Push
a neuron" never opens the form, by mouse or by Enter.
Steps: any page, click "+ edit", click "Push a neuron".
The popup closes and no form appears. Cause (read-only):
the document click-away closer kills the rowForm popup
that the same click just opened (the new popup does not
contain the detached click target). The demo-script guard
beat (add L11N1343 by hand) cannot run in the UI until
this is fixed. No workaround in the UI; the API path
works (above).

D2 (minor). Find while the inspector measures returns
409 with no rows. Steps: click a word, press Find before
the neuron list resolves. Server: 409 ("A run is in
progress"), console error, findOut stays empty. Same
class as B3, which disabled only Run.

D3 (minor, edge). A push that carries no reader words
silently skips the lift proof. Two ways to hit it:
pushing from the find list (`/api/find` returns no
`pushes_toward_top`), or pushing a neuron already in the
sentence (button disabled, run reuses the seeded setup).
The run forks normally but no proof line ever appears,
with no message. The scripted path (clear edits, open
the card, Push) is unaffected.

## Verify pass (2026-10-30, after 72b07ee)

Scripts `/tmp/day2-verify.js`, `/tmp/day2-l11.js`, shots
in `/tmp/smoke-day2-verify`. Live server, Llama 3.2 3B.

- D1 fixed. "+ edit → Push a neuron" opens the form, by
  mouse. The full L11 guard beat runs in the UI: add
  11/1343/2.0, Run → BLOCKED ("The refusal guard blocked
  L11N1343 +2.0. ... fell to 0% of its usual level on "a
  request for lock-picking instructions"; the limit is
  50%."). The named worst prompt varies run to run
  (lock-picking, forgery, fentanyl across runs).
- D2 fixed. Find during an inspect waits and returns 8
  rows; no 409 in the console.
- D3 fixed. The proof line renders on the scripted path
  (17550x again).
- L11 at 0.0 in the UI matches run 1 ("NO EDIT ... no
  new run), because dose 0 is a no-op the client
  short-circuits — it never reaches the guard. The API
  confirms 0.0 passes verified (min_ratio 1.0). L11 at
  0.5 and 0.25 both block on the full confirm
  (min_ratio 0.24/0.31), so there is no nonzero L11 dose
  that runs and passes; the demo-script "passes" beat
  now says what the screen shows. B6 persists.
- Anomaly (unreproduced, one run): the first verify run
  showed an L11N1343 row in the sentence before the add
  form was submitted (shot v01). The two clean re-runs
  (and a minimal repro) show the expected L19 sentence
  at boot. Not pursued further.
