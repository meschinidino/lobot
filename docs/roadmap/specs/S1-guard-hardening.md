# S1 Guard hardening and the demo guard

Owner: backend agent (code), then one model slot (measurement).
Lane: no-model, then model. Depends on: nothing. Time: one day.

## Goal

Make the refusal guard ready for beat 3 and honest to claim: it blocks the
weakening edits it should, on prompts it was **not** tuned on, it says why, and
the demo runs with it on.

## Read first

- `guard.py` (prefill screen, ratio limit 0.5, natural floor 0.2, full
  confirm), `scripts/gate_guard.py`, `tests/test_guard.py`,
  `docs/open-questions.md`.

## Deliverables

1. **Held-out check.** Split the refusal set and the known weakening edits
   into tuning and held-out parts (fixed seed, written to
   `store/guard/splits.json`). `gate_guard.py --held-out` reports misses and
   false blocks on the held-out part only.
2. **Late refusals.** The prefill screen measures first-token refusal. Add a
   second token pattern check for "Sure, ... I can't" style answers in the
   full-generation confirm, using the existing judge or the refusal markers.
   Test it.
3. **Combined edits.** The guard checks the whole edit set, not each edit
   alone. Add a test where two edits each pass and together block.
4. **Demo default.** `LOBOT_GUARD_DEFAULT=1` makes `"guard"` default to true
   in `/api/run`, `/api/update` and `/api/create`; `/api/state` reports
   `guard_default`. Guard dicts carry `limit` and `floor` (contracts section 6).
5. **Plain reason.** Every blocked guard dict has `worst_prompt`, `min_ratio`,
   `limit`, `blocked_prompts`. No prompt text in the dict is harmful in a
   screen recording: keep a `display_prompt` field with a safe paraphrase for
   each refusal-set prompt (for example "a request for weapon instructions"),
   and the UI shows that, not the raw prompt.
6. `docs/api.md` and `docs/open-questions.md` updated.

## Tests

- `test_guard_default_env_on_and_off`.
- `test_blocked_dict_has_limit_floor_display_prompt`.
- `test_combined_edits_block_when_each_passes_alone` (fake engine).
- `test_late_refusal_caught_by_full_confirm` (fake generation "Sure! ... I can't help").
- `test_held_out_split_is_disjoint_and_fixed`.

## Acceptance checks

```bash
uv run python -m unittest discover -s tests -t .
uv run python scripts/gate_guard.py --model meta-llama/Llama-3.2-3B-Instruct --held-out
```
Report on the held-out part: misses (target 0), false blocks (report the
number), mean guard time per run (target under 5 s with the cache cold, under
0.2 s warm).

## Done means

Code, tests, the held-out numbers, the demo default, and a one-paragraph claim
the pitch may use, worded exactly as the numbers allow.

## Out of scope

New risk categories (honesty, privacy): later.md.

## Risks

A miss on the held-out part. Do not tune on the held-out part to fix it; report
it, then widen the tuning set and re-split with a new seed, and say so.
