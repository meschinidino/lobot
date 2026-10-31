# CH Chat with live dials

Owner: backend agent. Lane: no-model for the code and tests; one
short model slot for the replay + dial-equality checks. Depends
on: D1 (dial math), U3 (tree dials). Time: one day.

## Goal

Multi-turn chat where edits persist across turns (exists, E2) and
read dials stream per word across turns (new). A chat replays
exactly: the same messages and edits reproduce the same replies
and dials. Chat dials match `/api/run` dials token for token.

Acceptance (from `later.md`): 10-turn chat replays exactly;
dials per token match /api/run reads.

## Read first

- `src/model_lobotomy/server.py`: `_chat_stream`, `_dial_spec`,
  `_wrap_chat`, `_run_stream` (the dial tap path).
- `src/model_lobotomy/engine.py`: `run_stream` (single-dial taps).
- `docs/roadmap/specs/D1-read-dials.md` and `reports/D1.md`.

## Deliverables

- `POST /api/chat/start` gains `dial_traits: [family]` (0-4;
  default empty). Unknown trait: 404 as today.
- Chat `token` events carry `dial: {trait: value}` per trait
  (a trait whose tap lagged is absent for that token, like runs).
  `done.result` gains `dials: {trait: [{text, value}]}` for the
  turn; `GET /api/chat/{id}` returns turns with their dial lists.
- `Engine.run_stream` takes `dials: [spec]` (list; `dial` stays
  as the one-spec shorthand). One tap per layer serves every
  spec on that layer.
- Replay as a feature: `GET /api/chat/{id}/transcript` (messages
  plus the per-turn rows/trait/dose/dial_traits) and `POST
  /api/chat/replay {"transcript": ...}` (re-runs every turn in a
  fresh session, returns `{"match": bool, "turns": [{turn,
  "match", "reply_match", "dials_match"}]}`). Mismatch details
  name the first divergent turn.
- `scripts/check_chat_replay.py`: drives 10 turns (edits change
  mid-chat, two dial traits), replays, prints match per turn;
  then runs the same first turn through `/api/run` and compares
  dial values token for token. Exit 0 on full match.
- `tests/test_chat_dials.py` (no model; scripted engine).
- `docs/api.md` updated. Report `reports/CH.md`.

## Method notes

- Dial specs resolve once per turn (anchors cached in
  `dial.json` as today). A trait whose knob vanishes mid-chat
  errors the turn like a vanished knob today (404 text in an
  `error` event, no partial dials).
- Single-message chat wrap must equal the run wrap so turn-1
  dials can match `/api/run`: both go through one helper that
  applies the chat template to one user turn. Multi-turn wraps
  the history as today.
- Determinism: greedy, fixed seed, same wrap, same edits. Replay
  holds the engine lock per turn (one at a time, like runs).
- Transcript replay pins the transcript's model_id: a replay on
  another active model is 409, like the session pin today.

## Tests (no model; scripted engine)

- `test_token_events_carry_many_dials` (two traits, one token).
- `test_lagging_tap_omits_that_trait_only`.
- `test_done_result_carries_dial_lists`.
- `test_history_returns_dial_lists`.
- `test_transcript_round_trip_matches` (scripted deterministic
  engine: replay matches all turns).
- `test_replay_flags_divergence` (scripted engine changes one
  turn: match false, turn named).
- `test_single_wrap_equals_run_wrap` (fake tokenizer records
  the template call: same messages both paths).
- `test_unknown_dial_trait_404`, `test_too_many_dial_traits_400`.
- Route 409/503 follow the existing chat tests.

## Acceptance checks

```bash
uv run python -m unittest tests.test_chat_dials -v
uv run python -m unittest discover -s tests -t .      # full suite passes
uv run python scripts/check_chat_replay.py            # exit 0, 10/10 match, dials equal
```

The script needs weights (model lane). It prints the per-turn
match table and the max abs dial difference vs `/api/run`
(must be exactly 0.0).

## Done means

All of the above, plus a report with the exact script output.
`docs/api.md` documents `dial_traits`, the dial events, and the
two replay routes.

## Out of scope

Persisting chats across restarts (memory only, as today); more
than 4 dial traits; any UI (U3 owns the dial tree; it reads the
new fields).

## Risks

- Tokenizer wrap differs between run and chat paths (extra
  tokens shift every dial). The wrap test catches it without
  weights; fix by sharing the helper.
- Floating-point nondeterminism across runs (MPS reductions).
  If replay mismatches by epsilon, compare with exact equality
  first and report; do not loosen to "close enough" silently.
