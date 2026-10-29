# T9 verdict: PASS

Integration lane `tests/test_steering_loop.py` (D5: slow, opt-in per run).
Run: `MODEL_LOBOTOMY_INTEGRATION=1 uv run python -m unittest
tests.test_steering_loop -v` (rung 1, Qwen/Qwen3.5-4B fp16 MPS, calibration
cache hit on the committed `calibration.json` via a temp copy — committed
file untouched).

Verbatim (from `logs/t9_integration.log`, git-ignored run receipt):

```text
Ran 5 tests in 18.435s

OK
```

- test_canary_hook_fires ... ok
- test_steered_hidden_states_differ ... ok
- test_dose_zero_logits_bit_identical ... ok
- test_dose_zero_generation_exact ... ok (32-token greedy, token-for-token)
- test_baseline_greedy_deterministic ... ok (D7 baseline-vs-baseline control)

Failures: none.

Opt-in gate verified: without `MODEL_LOBOTOMY_INTEGRATION=1` all 5 skip
(`OK (skipped=5)`); full discover stays `Ran 71 tests ... OK (skipped=5,
expected failures=1)` — the 66 T8 unit tests plus the 1 pinned
expected-failure (T8 calibration-pooling bug, engine owner lane, untouched).
