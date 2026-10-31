# Agent boot runbook: run the app for the user

Use this when the user asks to run, try, or open the app. For
human-facing guides, see [replay mode](replay-mode.md) and [live
mode](live-mode.md).

## Before you start

1. Confirm the branch holds the code to run (`git branch
   --show-current`). Uncommitted changes run as they stand; say so
   if the tree is dirty.
2. Check the port is free. `curl` exit `000` means free:
   `curl -s -m 5 -o /dev/null -w "%{http_code}"
   http://127.0.0.1:7860/` — treat another process on the port as
   the user's; use `PORT=` for yours instead.
3. Loading weights needs the model lane. Read
   [model-lane.md](roadmap/model-lane.md): if another agent holds
   it, wait or stay in replay mode. Otherwise append a `start`
   line before the first run.

## Start commands

Live (weights load on boot or first run):

```sh
uv run python -m model_lobotomy.server >>/tmp/lobot-live.log 2>&1
```

Replay (no weights; only recorded setups answer):

```sh
MODEL_LOBOTOMY_REPLAY=1 uv run python -m model_lobotomy.server >>/tmp/lobot-replay.log 2>&1
```

Layout only (every run returns 503):

```sh
MODEL_LOBOTOMY_NO_MODEL=1 uv run python -m model_lobotomy.server >>/tmp/lobot-nomodel.log 2>&1
```

Always log to a file. Never pipe server stdout through `head`:
`head` exits, stdout closes, and later runs fail with
`[Errno 32] Broken pipe`. This has bitten once already.

## Verify before you claim it

1. State: `curl -s http://127.0.0.1:7860/api/state` shows
   `replay`, `active_model`, and `warm`.
2. Page: `curl -s -o /dev/null -w "%{http_code}"
   http://127.0.0.1:7860/` returns 200.
3. Re-check immediately before telling the user it runs. A stale
   check is not a claim.

## Lifetime warning

Every process you start dies with your session, and macOS has no
`setsid` to detach with. Always hand the user the relaunch
command for their own terminal along with the link.

## While it runs

- One run at a time. HTTP 409 means busy: wait or POST
  `/api/stop`. If the lock survives Stop for minutes, the run is
  stuck; restart the server (a stuck lock past Stop happened
  once).
- Default model is Qwen (public, about 9 GB on first run).
  Llama is gated: it needs the Hugging Face license plus `uv
  run hf auth login` once per machine.
- Switching models unloads first (`Engine.load` always unloads),
  so running Llama shuts Qwen down by itself.
- In replay, only recorded setups answer (see the replay guide
  table). Anything else gets 404; behaviors load real weights
  instead of replaying.

## Debugging for an agent

Reproduce through the API before you theorize. `curl`
`/api/state` and the failing route with a minimal body. Quote
the verbatim error plus the command that produced it.

Read the server log tail first. It shows tracebacks, the
request log with status codes, and weight-load progress. The
log path is the file from the start command.

Common failures and their causes:

- 404 "not recorded": replay mismatch. Compare model, prompt
  text, and rows against `web/sweep*.json` byte for byte.
- 409 "run in progress": the lock is busy. Wait or POST
  `/api/stop`. A lock that survives Stop for minutes is stuck;
  restart the server.
- 503 "Weights are not loaded": `NO_MODEL` is set, or a replay
  route needs the model (neurons, Find, Lift). Restart in the
  right mode.
- `[Errno 32] Broken pipe` inside a run: server stdout died
  (usually a `head` pipe). Restart logging to a file.
- Gated-model error: Llama needs the license plus `uv run hf
  auth login`.
- Stale UI labels: hard-refresh the page; the app re-reads
  state after every run.

Code map: routes in `src/model_lobotomy/server.py`, weights
and hooks in `src/model_lobotomy/engine.py`, math in
`src/model_lobotomy/steer.py`, Create in
`src/model_lobotomy/create_job.py` plus
`src/model_lobotomy/trait_eval.py`. Tests mirror modules in
`tests/`.

Scope: the UI lane owns `web/`. Do not edit it unless the user
explicitly asks. Backend fixes need the suite green before any
claim.

## Rules that still bind

- Never push to `main` directly; `main` moves only by pull
  request from `develop`. Never force-push.
- Work on `feature/*` branches cut from `develop`.
- Suite: `uv run python -m unittest discover -s tests -t .`
  (339 pass, 5 pre-existing weight skips).
