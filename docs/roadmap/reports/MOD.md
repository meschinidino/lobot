# MOD Modularity: report

Spec: [MOD-modularity.md](../specs/MOD-modularity.md). One phase per
pull request, in spec order. No behavior change in any phase.

## Phase 1: Settings (this PR)

- New module: `src/model_lobotomy/config.py`.
- It holds one frozen `Settings` object and `ENV_VARS`.
- `get_settings()` builds a fresh object on every call.
- Tests that patch the environment keep the old behavior.
- Every `os.environ` read in `src/model_lobotomy/` moved there.
- Only `config.py` reads the environment in `src/` now.
- A test checks this rule.
- `llama_backend.CreateDaemon` passes its quant to `resolve_gguf`.
- It no longer writes `LOBOT_LLAMA_QUANT` to the environment.
- `docs/reference/configuration.md` did not change.
- A test checks the doc and `ENV_VARS` list the same variables.
- No route, file format, number or screen changed.

Tests:

- Before: `Ran 696 tests, OK (skipped=5)` on `develop`.
- After: `Ran 703 tests, OK (skipped=5)`.
- New tests: 7 in `tests/test_config.py`.

Parity:

- Replay parity: not needed in this phase. No server route changed.
- Live parity: not run. The spec reserves one model slot for the end.
- Tested with fakes and unit tests only. No weights loaded.

Known gaps:

- `scripts/` still reads the environment in two stub scripts.
  Later phases touch scripts and will move those reads.
- New port settings (`LOBOT_GUARD`, `LOBOT_JUDGE` and others) do not
  exist yet. Each later phase adds its own setting.

## Phases 2 to 10

Not started. Each phase gets its own branch and pull request.
Phase order: job storage, library code out of scripts, judge port,
guard port, direction and dose-rule ports, Create stages, server
routers, frontend modules, docs.
