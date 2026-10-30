# lobot backend API (waves 0+1)

Base URL: `http://127.0.0.1:7860` (or `$PORT`). All POST bodies are JSON.
Cross-origin POSTs are rejected with 403. A browser POST must carry an
`Origin` equal to the server's own origin, on a loopback host (127.0.0.1,
localhost or ::1), so a DNS-rebound domain is rejected too. Requests
without an `Origin` header (curl, scripts) are allowed. One generation runs at a
time: a second run while one holds the engine lock gets 409.

Conventions: the refusal guard is opt-in until release (`guard:
true` to enforce; default off). Every run still checks and records
`would_block`; `research_mode: true` also never blocks and wins over
`guard`. The check runs before any generation (`guard` stage); a
block ends the stream with `blocked`. Guard dicts carry `enforced`
(true only when a block could land).

## GET /api/state

Models, active model, warm status, dims, preset rows. No model needed.

## POST /api/model

`{"model_id": "..."}` — switch the active model (unloads; next run
reloads). 400 unknown model, 409 run in progress.

## POST /api/run → NDJSON stages

One before/after run over neuron rows.

Request: `{"prompt": str, "rows": [{"layer", "neuron", "dose"}],
"guard": false, "research_mode": false}`. Dose range and max rows
come from `/api/state` (`dose_min`, `dose_max`, `max_rows`).

Events (one JSON object per line):

| stage | fields |
|---|---|
| `weights` | — (model loads here) |
| `guard` | — (refusal check passed; see `blocked`) |
| `baseline` | — (clean generation running) |
| `steered` | — (steered generation running) |
| `stopped` | `after`: stage the Stop landed after |
| `blocked` | `guard`: `{blocked, would_block, enforced, verified, min_ratio, worst_prompt, blocked_prompts}` (`verified`: the allow was confirmed by full generation + judge; prefill-only blocks show `verified: false`) |
| `error` | `message` |
| `complete` | `result`: `{model_id, prompt, rows, baseline: {text, tokens}, steered: {text, tokens}, split: {token, base_char, steered_char} \| null, seconds, cache_hit, guard, recipe, post}` (`guard`: the recorded verdict dict, `blocked: false` here) |

Errors: 400 empty prompt/recipe/invalid row, 409 busy, 503 NO_MODEL.

Replay mode (`MODEL_LOBOTOMY_REPLAY=1`) loads no weights. `/api/run`
answers only setups recorded in `web/example.json` and `web/sweep*.json`
(same model, prompt and rows, in any order) with the same stage events
and Stop, plus `replay: true` in the result; any other setup gets 404.
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
"research_mode": false}`. `description` must match a validated
family name (case-free).

Errors: 422 description is "not supported yet" (unknown, or family
without a validated judge), 503 NO_MODEL.

## GET /api/create/{job_id}

Job status (reattach-safe: served from disk when the job is not live
in this process). 404 unknown job.

Status shape: `{"id", "trait", "seed", "model_id", "revision",
"research_mode", "guard_opt_in", "status", "stage", "progress":
{"stage", "done", "total"}, "verdict", "frozen": {"layer", "dose",
...} | {}, "eligibility", "guard", "error", "created", "updated"}`.
`guard_opt_in` echoes the request flag; the `guard` dict is the
recorded verdict (`{blocked, would_block, enforced, ...}`).

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

## Frontend loop (happy path)

1. `GET /api/families` → offer validated traits.
2. `POST /api/create {"description"}` → poll `GET
   /api/create/{job_id}` until `done` (`verdict: knob`).
3. `POST /api/update {"prompt", "trait", "doses"}` → render
   `columns`, append `token` events per `col`, show `complete`.
4. `POST /api/create/{job_id}/stop` or `/api/stop` to cancel.
