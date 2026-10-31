# lobot backend API (waves 0+1)

Base URL: `http://127.0.0.1:7860` (or `$PORT`). All POST bodies are JSON.
Cross-origin POSTs are rejected with 403. A browser POST must carry an
`Origin` equal to the server's own origin, on a loopback host (127.0.0.1,
localhost or ::1), so a DNS-rebound domain is rejected too. Requests
without an `Origin` header (curl, scripts) are allowed. One generation runs at a
time: a second run while one holds the engine lock gets 409.

Conventions: the refusal guard is opt-in until release (`guard:
true` to enforce; default off, or on when the server runs with
`LOBOT_GUARD_DEFAULT=1`). Every run still checks and records
`would_block`; `research_mode: true` also never blocks and wins over
`guard`. The check runs before any generation (`guard` stage); a
block ends the stream with `blocked`. Guard dicts carry `enforced`
(true only when a block could land), the `limit` (0.5) and `floor`
(0.2) in force, and `display_prompt` (a safe paraphrase of
`worst_prompt` for the UI to show).

## GET /api/state

Models, active model, warm status, dims, preset rows, `guard_default`
(the `guard` default from `LOBOT_GUARD_DEFAULT`). No model needed.

## POST /api/model

`{"model_id": "..."}` — switch the active model (unloads; next run
reloads). 400 unknown model, 409 run in progress.

## POST /api/run → NDJSON stages

One before/after run over neuron rows.

Request: `{"prompt": str, "rows": [{"layer", "neuron", "dose"}],
"guard": false, "research_mode": false, "dial_trait": null}`. Dose
range and max rows come from `/api/state` (`dose_min`, `dose_max`,
`max_rows`). `guard` defaults to the state's `guard_default` when
omitted.

Events (one JSON object per line):

| stage | fields |
|---|---|
| `weights` | — (model loads here) |
| `guard` | — (refusal check passed; see `blocked`) |
| `baseline` | — (clean generation running) |
| `steered` | — (steered generation running) |
| `stopped` | `after`: stage the Stop landed after |
| `blocked` | `guard`: `{blocked, would_block, enforced, verified, min_ratio, worst_prompt, blocked_prompts, limit, floor, display_prompt}` (`verified`: the allow was confirmed by full generation + judge; prefill-only blocks show `verified: false`) |
| `token` | `text` (append), `dial`: `{trait: value}` (only with `dial_trait`, during `steered`) |
| `error` | `message` |
| `complete` | `result`: `{model_id, prompt, rows, baseline: {text, tokens}, steered: {text, tokens}, split: {token, base_char, steered_char} \| null, seconds, cache_hit, guard, dial, recipe, post}` (`guard`: the recorded verdict dict, `blocked: false` here; `dial`: `{trait, layer, neutral, trait_mean, baseline, steered}` or null, with one `{text, value}` per generated token of each pass) |

Errors: 400 empty prompt/recipe/invalid row, 409 busy, 503 NO_MODEL.

Read dial: `dial_trait: "<family>"` resolves the newest finished
Create job for that trait and streams one `token` event per steered
token carrying `dial: {trait: value}`. Scale: 0 = the trait's
neutral-template mean projection, 1 = its trait-template mean
(anchors in `complete.result.dial`; cached per job in
`jobs/{id}/trait/dial.json`). Tokens whose tap entry is missing
ship without `dial`. The baseline pass gets the same `Read` edit
(it never changes the baseline's tokens). `complete.result.dial`
adds `baseline` and `steered`: one `{text, value}` per generated
token of each pass (`value` is null where the tap lagged; the
texts join to the pass's text; the steered list equals the
streamed values). Unknown trait: `error` event (create it first).

Replay mode (`MODEL_LOBOTOMY_REPLAY=1`) loads no weights. `/api/run`
answers only setups recorded in `web/example.json` and `web/sweep*.json`
(same model, prompt and rows, in any order; a setup with every dose at 0 plays any recording of that prompt as its baseline) with the same stage events
and Stop, plus `replay: true` in the result; any other setup gets 404.
`dial` is the recorded dial when the recording has one, else null.
`/api/state` reports `replay`.

## POST /api/stop

`{}` — set the Stop flag for the active `/api/run` or `/api/update`
stream. Returns `{"stopping": bool}` (false when idle).

## GET /api/families

`{"families": [{"name", "aliases", "pos", "neg", "judge",
"hand_validated", "validation": {"hand_labels", "agreement", "note"}}]}`.
Create accepts validated families only.

## POST /api/create → job

Start a staged trait run (extract → tune → read_gate → final →
baseline → verdict). Returns `{"job_id", ...status}`.

Request: `{"description": str, "seed": 0, "guard": false,
"research_mode": false, "schedule": "full"}`. `description` must
match a validated family name (case-free). `guard` defaults to
the state's `guard_default` when omitted. `schedule` is `full`
(the reference pipeline; default) or `fast` (fewer prompts,
shorter generations, early stop; preliminary — its verdicts
currently disagree with full, see `roadmap/reports/C2.md`).

Errors: 422 description is "not supported yet" (unknown, or family
without a validated judge), 503 NO_MODEL.

## GET /api/create/{job_id}

Job status (reattach-safe: served from disk when the job is not live
in this process). 404 unknown job.

Status shape: `{"id", "trait", "seed", "model_id", "revision",
"research_mode", "guard_opt_in", "schedule", "status", "stage",
"progress": {"stage", "done", "total}, "verdict", "frozen":
{"layer", "dose", ...} | {}, "eligibility", "guard", "error",
"created", "updated"}`. The trait report records the schedule,
per-stage seconds (`timings`), early-stop decisions
(`early_stop`), and base-cache hits.
`guard_opt_in` echoes the request flag; the `guard` dict is the
recorded verdict (`{blocked, would_block, enforced, verified,
min_ratio, worst_prompt, blocked_prompts, limit, floor,
display_prompt, research_mode}`).

`status`: `queued` (waiting for the engine lock) | `running` |
`done` | `stopped` | `failed` (`error` carries the traceback tail).
`verdict`: `knob` | `no_knob` | `blocked` (guard refused the frozen
direction; only when enforced). The full trait report is
`jobs/{id}/trait/report.json`;
the frozen direction is `jobs/{id}/trait/direction.safetensors`
(`{"direction": tensor}`).

## POST /api/create/{job_id}/stop

Stop a queued/running job (the pipeline raises at generation
granularity). Returns the status. 404 unknown job.

## GET /api/jobs

`{"jobs": [status, ...]}` — every job on disk, oldest first.

## GET /api/traits (E1 evidence)

`{"traits": [item, ...]}` — one item per finished (`done`) Create
job with a readable report, newest first. No model needed.
`?model_id=...` keeps one model only.

Item: `{"id", "family", "model_id", "version", "verdict", "layer",
"dose", "created", "cards", "guard", "patch"}`. `version` counts
finished runs per family and model, oldest first (v1, v2, ...).
`layer`/`dose` are the frozen choice (null without one). `cards`
holds `update_up`, `update_down`, `delete`, `prompt_baseline`,
each `{"passed", "trait_mean", "trait_ci"}` or null. `guard` is
`{"blocked", "would_block", "verified"}` or null. `patch` is the
download URL, or null without a saved patch.

## GET /api/traits/{id}/report (E1 evidence)

The job's `report.json` as is. 404 unknown id or no report.

## GET /api/traits/{id}/report.md (E1 evidence)

The job's `report.md` as `text/markdown`. 404 unknown id or no
report.

## GET /api/traits/{id}/patch (E1 evidence)

The job's `knob.lobot` patch as `application/zip`, download name
`<family>-v<version>.lobot`. 404 unknown id or no patch. The file
loads with `patch.load_patch`.

## POST /api/update → NDJSON tokens

Vector Update filmstrip: one batched generate across doses (E6 hook;
falls back to sequential singles and says so). Either a trait (uses
the newest finished Create job's saved direction) or an explicit
layer/direction/scale.

Request:

```json
{
  "prompt": "Say hi.",
  "trait": "blunt",
  "layer": 5,
  "direction": [0.01, -0.02],
  "scale": 2.5,
  "doses": [0.0, 0.5, 1.0],
  "max_new_tokens": 60,
  "guard": false,
  "research_mode": false
}
```

Rules: `trait` XOR (`layer` + `direction` + `scale`). 1–8 doses.
`direction` must have exactly `hidden` dims (see `/api/state`;
3072 on Llama 3.2 3B). With a trait, `scale` is the calibrated
layer median norm. `max_new_tokens` is clamped to the server cap.
`guard` defaults to the state's `guard_default` when omitted.

Events:

| stage | fields |
|---|---|
| `weights` | — (model loads + layer medians calibrate here) |
| `guard` | — (strongest-dose check passed) |
| `columns` | `doses`, `scale`, `guard` (verdict dict, as in `/api/run`) |
| `token` | `col` (index into `doses`), `text` (append), `dose`, `fallback` (true only on the singles fallback, where each column arrives as one full-text event at completion) |
| `stopped` | `after`: `weights` or `columns` |
| `blocked` | `guard` (verdict dict) |
| `error` | `message` |
| `complete` | `result`: `{model_id, prompt, layer, scale, doses, columns: [{dose, text, tokens}], fallback_used, guard, seconds, cache_hit}` |

Concatenating each column's `token` texts in arrival order reproduces
its `complete` text. Columns interleave; order by `col`.

Errors: 400 (empty prompt, 0 or 9+ doses, trait+explicit mixed,
missing explicit field, layer/direction shape), 404 (no finished
Create job with a saved direction for the trait), 409 busy, 503
NO_MODEL.

## POST /api/neurons (R1 click-a-word, R2 reply words)

Top neurons at one prompt or reply token, measured with the run's
edits on.

Request: `{"prompt": str, "rows": [...], "trait": null, "dose":
null, "token_index": int, "top_k": 8, "window": 4, "guard": false,
"research_mode": false, "reply": null}`. `dose` defaults to the
knob's frozen dose; `top_k` clamps to 1–32, `window` to 0–16
tokens. When `reply` holds a generated answer, it is appended
after the wrapped prompt and `token_index` may address its
tokens. Base sends `rows: []` with the baseline text; steered
sends the run's rows with the steered text. Instead of
`token_index`, a request may give `reply_char` (a character offset
into `reply`): the server uses the reply token that covers it (a
space belongs to the next word's token). Give exactly one of the two;
the response's `token_index` is the one used.

Response (one JSON object, not a stream):

```json
{
  "blocked": false,
  "model_id": "Qwen/Qwen3.5-4B",
  "prompt": "Say hi.",
  "tokens": ["Say", " hi", "."],
  "token_index": 1,
  "neurons": [
    {"layer": 14, "neuron": 8949, "activation": 9.2,
     "top_contexts": [
       {"position": 1, "activation": 9.2, "text": "ay hi."}]}
  ]
}
```

`tokens` is the wrapped prompt's token split (plus the reply
split when `reply` is set); `token_index` addresses it starting
at 1 (position 0 is the BOS slot and has no captured row).
`reply_start` is the index of the first reply token, or null
without a reply. `neurons` ranks every layer's MLP-input
activations
at that token by absolute value. Each entry's `top_contexts` are
that neuron's 3 highest-activating positions in this run with
a ±`window` decoded span, and `pushes_toward_top` holds its top
3 pushed tokens (null without the reader). Guard blocks return
`{"blocked": true, "guard": {...}}` with no neurons.

Errors: 400 (empty prompt, too many rows, bad row, bad
`token_index` or `reply_char`, both or neither given), 404 (trait
without a finished Create job), 409 busy, 503 NO_MODEL or replay mode.

## GET /api/neuron/{layer}/{neuron} (R2 neuron card)

One neuron card: what it pushes toward plus cached contexts.

`GET /api/neuron/11/1343?k=12` → `{"model_id", "layer",
"neuron", "pushes_toward": {"up", "down", "coherence",
"layer_frac"}, "fires_on": [{"activation", "token", "context",
"mark"}], "fires_on_cache": {"built", "tokens"}}`. `k` clamps
to 1–50. `fires_on` is null and `built` is false when no cache
exists for the model. In replay mode the card comes from
`store/reader/<model-slug>/cards.json`.

Errors: 400 (neuron out of bounds), 404 (no recorded card in
replay mode), 409 busy, 503 NO_MODEL (or no cards file in
replay mode).

## GET /api/find (R2 find a neuron)

Neurons that push one word hardest.

`GET /api/find?word=pirate&layers=8-27&k=20` → `{"word",
"token_ids", "results": [{"layer", "neuron", "score",
"rank"}]}`. `layers` takes `all`, one layer (`14`), or an
inclusive range (`8-27`); it defaults to all layers. `k`
defaults to 20 (1–50). `rank` is the 1-based position of the
word in the neuron's own up list, or null past 100.

Errors: 400 (empty word, bad layer range, word with no
tokens), 409 busy, 503 NO_MODEL or replay mode.

## POST /api/lift (R2 steering lift)

How much steering one neuron lifts its predicted words.

Request: `{"layer", "neuron", "dose", "words": ["pirate"],
"prompts": null}`. `dose` stays in the state bounds;
`prompts: null` uses the 8 fixed lift prompts. Response:
`{"words", "base_p", "steered_p", "lift"}` (per-word next-token
means and their geometric-mean ratio).

Errors: 400 (neuron out of bounds, bad word), 422 (dose out
of range, no words), 409 busy, 503 NO_MODEL or replay mode.

## GET /api/features (F1 clean features)

Trained sparse autoencoders for the active model. No model needed.

`{"saes": [{"model_id", "layer", "features", "top_k", "live",
"dead", "mse"}]}` — one entry per trained layer, `mse` is the
final train MSE. Empty list when nothing is trained.

## GET /api/feature/{layer}/{index} (F1 feature card)

One SAE feature card: what it pushes toward plus cached contexts.

`GET /api/feature/12/4330?k=12` → `{"model_id", "layer",
"index", "live", "pushes_toward": {"up", "down", "coherence",
"layer_frac"}, "fires_on": [{"activation", "token", "context",
"mark"}], "fires_on_cache": {"built", "tokens"}}`. `live` is
false when the feature never fires in the cache. `k` clamps to
1–50.

Errors: 400 (layer or index out of bounds), 404 (no trained SAE
for the layer), 409 busy, 503 NO_MODEL or replay mode.

## POST /api/feature-lift (F1 feature steering lift)

How much steering one SAE feature lifts its predicted words.

Request: `{"layer", "index", "dose", "words": null,
"prompts": null}`. `words: null` uses the feature's top words
(the top-3 whole-word up tokens, else the top-1 up token);
`prompts: null` uses the 8 fixed lift prompts. The feature
steers as a vector edit with scale = the layer median norm.
Response: `{"words", "base_p", "steered_p", "lift"}`.

Errors: 400 (index out of bounds, bad word), 404 (no trained SAE
for the layer), 422 (dose out of range, no words), 409 busy, 503
NO_MODEL or replay mode.

## Chat (E2)

Multi-turn with edits that persist across turns. History lives
server-side (in memory; a restart clears it).

`POST /api/chat/start {"rows": [], "trait": null, "dose": null,
"max_new_tokens": 150, "guard": false, "research_mode": false}` →
`{"session_id", "model_id"}`. At most 16 sessions; the oldest
goes. Errors: 400 (too many rows, bad row), 404 (trait without a
finished Create job).

`POST /api/chat {"session_id", "message", "rows": null,
"max_new_tokens": null}` → NDJSON. `rows` replaces the session's
edits when present (null keeps them); the trait and dose persist.
Events:

| stage | fields |
|---|---|
| `token` | `text` (append) |
| `blocked` | `guard` (verdict dict; only when enforced) |
| `stopped` | `after`: `guard` or `chat` |
| `error` | `message` |
| `done` | `result`: `{session_id, turn, reply, tokens, seconds}` |

History caps at 40 turns (oldest pairs drop). Errors: 400 empty
message/bad rows, 404 unknown session or vanished trait knob, 409
busy or session pinned to another model, 503 NO_MODEL.

`GET /api/chat/{session_id}` → `{"session_id", "model_id",
"turns": [{"role", "content"}]}` for reattach. 404 unknown.

## Frontend loop (happy path)

1. `GET /api/families` → offer validated traits.
2. `POST /api/create {"description"}` → poll `GET
   /api/create/{job_id}` until `done` (`verdict: knob`).
3. `POST /api/update {"prompt", "trait", "doses"}` → render
   `columns`, append `token` events per `col`, show `complete`.
4. `POST /api/create/{job_id}/stop` or `/api/stop` to cancel.
