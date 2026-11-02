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

## Phase 2: Job storage

- New module: `src/model_lobotomy/jobs_store.py`.
- `JobPaths(job_id)` holds `.job_json`, `.report`, `.patch`,
  `.direction`, `.pareto` and `.repair(method)`.
- `WorkPaths(dir)` holds the same files for one trait work dir.
- `list_jobs()` and `import_run()` moved there.
- `job_import.py` now re-exports them. Old imports keep working.
- `create_job`, `followups`, `trait_eval`, `server` and
  `scripts/import_run.py` use these paths.
- New setting: `LOBOT_JOBS_DIR`. Default stays `<repo>/jobs`.
- No route, file format, number or screen changed.

Tests: before 703, after 706 (`OK, skipped=5`). New: 3 tests.
Parity: replay and live not run. No route changed its shape.
The 404 check for unknown jobs stays in `_trait_job_dir`.

## Phase 3: Library code out of scripts

- New module: `src/model_lobotomy/spike.py`.
- It holds `BEHAVIORS`, `separability`, `nll`, `best_dose`,
  `coherent_doses`, `_yes_no_ids` and the other spike measures.
- `trait_eval` and `trait_store` import it as a normal module.
- No `importlib` file loading stays in `src/`. A test checks this.
- `scripts/spike_knob.py` imports the functions back.
- Its `--self-test` still passes.

Tests: before 706, after 708 (`OK, skipped=5`). New: 2 tests.
Parity: replay and live not run. Numbers did not change.

## Phase 4: Judge port

- New package: `src/model_lobotomy/judges/__init__.py`.
- One interface: `score_many(pairs, label) -> list[float]`.
- The memo cache lives in `MemoJudge`. It dedupes pairs per run.
- `TorchLane` and `LlamaLane` score through it. They keep
  `judge_many` as the old name.
- The guard frozen judges, `refusal_judge_fn` and re-verify
  `judge_fn_for` go through `get_judge("local", ...)`.
- `MetaApiJudge` gains `score_many` from the `Judge` base.
- New setting: `LOBOT_JUDGE` (`local`, `meta_api`, `fake`).
- Contract tests run the same checks on fake and local judges.

Tests: before 708, after 714 (`OK, skipped=5`). New: 6 tests.
Parity: replay and live not run. Scores did not change.

## Phases 5 to 10

Not started. Each phase gets its own branch and pull request.
