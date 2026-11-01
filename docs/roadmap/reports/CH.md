# CH report: chat with live dials

Outcome: **ACCEPTANCE PASS.** 10-turn replay matches exactly;
chat dials equal `/api/run` dials token for token (max abs
diff 0.0).

## What changed

- `Engine.run_stream` takes `dials: [...]` (one tap per layer
  serves every spec on it; `dial` stays as the one-spec
  shorthand with identical output).
- `POST /api/chat/start` gains `dial_traits` (at most 4; each
  needs a finished knob, else 404).
- Chat `token` events carry `dial: {trait: value}`; a lagging
  tap omits only its trait. `done.result` gains per-turn
  `dials`; history gains `dial_traits` + `dial_turns`.
- Turn records (`message`, `rows`, `max_new_tokens`, `reply`,
  `dials`) persist per session (trimmed with history).
- `GET /api/chat/{id}/transcript` (replay spec) and `POST
  /api/chat/replay` (re-runs in a fresh session, per-turn
  match table with `dials_max_diff`; model-pinned, 409 else).
- `_dial_spec` reads run configs from the store (open C3
  families work as dials; legacy configs are byte-equal).
- `scripts/check_chat_replay.py` (the acceptance script).
- Tests: `tests/test_chat_dials.py`, 11 tests, no model.
- `docs/reference/api.md` + `contracts.md` (§10) updated.

## Test counts

- `tests.test_chat_dials`: 11 pass.
- Full suite: 428 pass, 5 skipped (pre-existing weight skips).

## Acceptance output (exact)

`uv run python scripts/check_chat_replay.py --model
meta-llama/Llama-3.2-3B-Instruct` (exit 0):

- turns 1-10: done (150 dial token events each; turn 8: 92,
  short reply).
- replay turns 1-10: match=True reply=True dials=True
  max_diff=0.0 err=None (rows change at turn 6 covered).
- run-vs-chat dials: shape_ok=True max_abs_diff=0.0.
- OVERALL: PASS.

## Known gaps

- Replay is exact within one loaded process (same kernels,
  same order). Across processes MPS drifts (see C2: judge
  mean 0.003, greedy flips) — a replay after a model reload
  may mismatch by epsilon. The route reports `dials_max_diff`
  instead of hiding it.
- Dials resolve per turn; a knob deleted mid-chat errors the
  turn (documented, tested shape only with fakes).
- 4 dial traits max; raising it is trivial if the UI needs
  more.

## UI lane needs

- Dial picker on chat start (at most 4 traits with knobs).
- Per-word dial coloring from `token.dial` (U3 owns the tree).
- Transcript view + replay button showing the match table.
