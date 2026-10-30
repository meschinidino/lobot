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
