# J1 Meta-model judge, offline

Owner: backend agent. Lane: no-model (needs network and an API key).
Depends on: nothing (decided 2026-10-30: offline Meta-API judge is allowed). Time: half a day.

## Goal

Let the trait evaluation use a judge served by the Meta Model API, run before
the demo (never live on stage), with the judge model id recorded in every
report. The local judge stays the default and the fallback.

## Read first

- `src/model_lobotomy/trait_eval.py` (judge prompt and scoring), `scripts/validate_judge.py`,
  `docs/archive/open-questions.md`.

## Step 0 (do first)

Check whether the Meta model API returns token log-probabilities. The method
and fallbacks are in `docs/roadmap/README.md`, "Next step". The local judge's
score is P(Yes) / (P(Yes) + P(No)); `MetaApiJudge` must produce the same kind
of score, or say plainly which fallback it uses.

## Deliverables

- A judge interface: `Judge.score(question, reply, rubric) -> float in [0, 1]`
  with two implementations, `LocalJudge` (current behavior, unchanged scores)
  and `MetaApiJudge` (HTTP client; key from env `META_API_KEY`; model id from
  env `LOBOT_JUDGE_MODEL`; timeouts, retries with backoff, no key in logs).
- `report.json` adds `judge: {"kind": "local"|"meta_api", "model": "<id>",
  "agreement": <float or null>}`.
- `scripts/validate_judge.py --judge meta_api` reports agreement with the hand
  labels (same table as the local judge).
- Docs: how to set the key; never commit it.

## Tests (no network)

- `test_local_judge_unchanged` (same scores as before on fixtures).
- `test_meta_judge_parses_score` with a mocked HTTP response.
- `test_meta_judge_retries_then_fails_cleanly` (mocked 429/500).
- `test_report_records_judge_kind_and_model`.
- `test_no_key_in_logs_or_report`.

## Acceptance checks

```bash
uv run python -m unittest discover -s tests -t .
META_API_KEY=... LOBOT_JUDGE_MODEL=... uv run python scripts/validate_judge.py --judge meta_api
```
Agreement with hand labels is reported next to the local judge's. Use the Meta
judge for shipped evidence only if its agreement is at least as good.

## Done means

Interface, client, report field, tests, the agreement table in the report.

## Out of scope

Any live API call during the demo.
