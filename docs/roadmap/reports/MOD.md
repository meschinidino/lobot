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

Tests: before 708, after 715 (`OK, skipped=5`). New: 7 tests.
Parity: replay and live not run. Scores did not change.

## Phase 5: Guard port

- New package: `src/model_lobotomy/guards/__init__.py`.
- Port verdict: `GuardPortVerdict` with `blocked`, `would_block`,
  `verified`, `enforced`, `research_mode` and `screens`.
- Each screen holds `flagged_prompts`, `measure_text` and `measure`.
  Guard numbers (`min_ratio`, `limit`) live in `measure`.
- `ScreensV2Guard` wraps today's guard. `screen` is fast.
  `confirm` is full generation.
- `NoopGuard` is test-only. It proves a swap works.
- Callers use `get_guard()`. No `Guard()` stays in `src/` outside
  the port. `followups` uses public `refusal_entry` and
  `SCREEN_DISPLAY`. It no longer touches private members.
- Old `verdict_dict` did not change. Reports keep their shape.
- New setting: `LOBOT_GUARD` (`screens-v2`, `noop`).

Tests: before 715, after 719 (`OK, skipped=5`). New: 4 tests.
Parity: replay and live not run. Verdict numbers did not change.

## Phase 6: Direction and dose-rule ports

- New modules: `directions/__init__.py` and `dose_rules.py`.
- `DiffMeanDirection` is the default direction method.
- `NllFluencyRule` is the default dose rule.
- It holds `name` and `params`. The report writes both into
  `tuning.rule`, with the same keys as before.
- `tune_layer_dose` takes a rule object. With no rule it uses
  the default, so old calls keep their numbers.
- `trait_eval.coherent_doses` and `best_dose` delegate to it.
- New settings: `LOBOT_DIRECTION`, `LOBOT_DOSE_RULE`.
- A `fake` dose rule exists for tests.

Tests: before 719, after 723 (`OK, skipped=5`). New: 4 tests.
Parity: replay and live not run. Numbers did not change.

## Phase 7: Create stages

- New modules: `create/__init__.py`, `create/stages.py`.
- The stage order lives there, with each stage's UI label.
- `run_trait` now loops over that order. Each stage is one
  function that takes and returns a shared `ctx` dict.
- The early `no_knob` exit sets `ctx["stop"]`.
- `create_job.STAGES` reads the same order.
- A test checks the UI list in `web/index.html` matches.
- No new route. The spec marks its route idea as an example.
  The test reads the file instead, so no route changes.

Tests: before 723, after 726 (`OK, skipped=5`). New: 3 tests.
Parity: replay and live not run. Stage order did not change.
Known gap: the `pareto` stage still runs inside `baseline`,
as before. Its stage entry is a no-op placeholder in the loop.

## Phases 8 to 10

Not started. Each phase gets its own branch and pull request.
