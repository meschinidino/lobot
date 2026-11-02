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

## Phase 8: Server routers

- `server.py` drops from 3,133 to 1,940 lines.
- Routes move to `routers/`: run, create, evidence, drafts_agent,
  chat, install_verify, reader_features. 51 routes, same paths.
- Shared state moves into one `AppState` on `api.state`.
  Its properties delegate to the module globals, so tests that
  set `server._runner` keep working.
- `_ndjson` moves to `streaming.py`. Every stream still uses it.
- Router helpers read patched state live. Types are injected
  once so FastAPI resolves the same annotations.

Tests: before 726, after 729 (`OK, skipped=5`). New: 3 tests.
Parity: replay and live not run in this phase. Same routes.

## Phase 9: Frontend modules

- CSS moves to `web/css/app.css`. It is byte-identical.
- JS moves to native ES modules, no build step:
  `js/api.js`, `js/state.js`, `js/components/` (dial, chart,
  legend, strip) and `js/screens/` (workbench, evidence, library,
  new-behavior, agent, labeling, chat, run, share, pages).
- `web/index.html` now loads `css/app.css` and `js/app.js`.
- Static parity: the split JS has the same lines as the old
  inline script. Only `export` prefixes differ.
- Replay check: the replay server returns 200 for `/`,
  the CSS, the modules, `/api/state` and `/api/traits`.
- No screen changed its text, look or behavior by design.

Tests: before 729, after 732 (`OK, skipped=5`). New: 3 tests.
Live parity: not run. Browser DOM parity was not run: no
browser tool is wired in this session. Static parity and the
replay HTTP check stand in for it. They are not a browser test.

## Phase 10: Docs and swap demo

- `AGENTS.md` gains a "How to swap a part" section.
  It gives one worked example per port.
- `docs/guides/app.md` notes where each screen's code lives.
- Swap demo (unit, fake engine, no weights):
  `tests/test_swap_demo.py` runs Create end to end with
  `LOBOT_GUARD=noop`, `LOBOT_JUDGE=fake` and
  `LOBOT_DOSE_RULE=fake`. The report names the swapped parts
  in `ports` and `tuning.rule`.
- The same file checks acceptance 4: no `Guard(` build, no
  environment read and no hand-built jobs path in `src/`
  outside the port modules.
- The demo found three latent stage bugs from phase 7
  (missing `model_id`, `seed` and `ext`/`cfg` in stage ctx).
  They are fixed in this phase. The full suite had missed
  them: no fake-engine Create ran end to end before.

Tests: before 732, after 734 (`OK, skipped=5`). New: 2 tests.

## Live parity (once, at the end, model slot)

One full-schedule torch Create for pirate, seed 0.
Run: `logs/mod_live_pirate` (3,343.9 s, exit 0).

| Field | MOD live | HEADREF (Oct 31) | FD llama (Nov 1) |
|---|---|---|---|
| verdict | knob | knob | knob |
| frozen | L16 @ 0.5, gain 0.241 | L17 @ 1.5, gain 0.821 | L16 @ 0.5, gain 0.239 |
| eligibility | read + update_up, knob | read, knob | read + update_up, knob |
| read gate | 1.0, margin 0.5 | - | 1.0, margin 0.5 |
| guard blocked | False | - | False |

The HEADREF row does not match on layer, dose or eligibility.
Cause: HEADREF predates the FD dose rule (commit `0195717`).
FD changed the dose rule to `nll+fluency-v1` after HEADREF ran.
MOD did not change that rule. It only moved it behind a port.
Against the FD reference, the live run matches: same verdict,
layer, dose and eligibility. Tuning gain differs by 0.002.
Report cards pass and fail on the same verbs. Each trait mean
lies inside the other run's interval (update_up: MOD 0.371,
FD 0.333; intervals overlap on [0.223, 0.484]).

So: live parity holds against the current reference.
The spec's HEADREF pointer is stale. It needs a new torch
reference run after FD. That rerun is out of MOD scope.

Replay parity (phase 9): static JS parity plus replay HTTP
checks. Browser DOM parity was not run. No browser tool is
wired in this session.

## Fix 1: Pareto stage (stacked PR 1)

- `_stage_pareto` now runs `pareto_mod.run_pareto`.
- It reads `ctx["base"]` from the baseline stage.
- It emits `pareto` progress and writes `timings["pareto"]`.
- `_stage_baseline` no longer runs the prompt comparison.
- The stage comment that called it a placeholder is gone.
- `pyflakes` is a dev dependency (`pyproject.toml`, `uv.lock`).
- The pyflakes unit test lands with Fix 3.
- Fix 3 removes the router undefined names (more than 500 today).
- A clean full pyflakes run is not possible before Fix 3.
- Tests: `Ran 737 tests, OK (skipped=5)`. Before: 736 plus this PR's 1 new test in `tests/test_create_stages.py` (the count also reflects PR #31 tests already on develop).
- Replay: not run. No route or screen changed.
- Browser: not run. This PR touches no `web/` or router file.
- Live weights: not run. Unit tests and fakes only.

## Fix 2: Guard port callers (stacked PR 2)

- Every caller in `server.py`, `combine.py`, `trait_eval.py` and
  `followups.py` uses `get_guard()` plus `screen()` or `confirm()`.
- Scripts `gate_guard.py`, `parity_llama.py` and `lobot_registry.py`
  (file: `scripts/lobot_registry.py`) use the same port.
- `GuardPortVerdict` carries `report` and `screen_reports`.
- Those fields hold the old JSON shapes, so files do not change.
- `confirm(screens=[...])` runs full generation on named screens.
- `confirm(confirmed=True)` runs the always-full confirmed check.
- The forwarded `check*` methods on `ScreensV2Guard` are gone.
- The old-style methods on `NoopGuard` are gone.
- New tests: `tests/test_guard_swap_paths.py` (6 tests).
- With `LOBOT_GUARD=noop`, they run run/chat/removal/verify (all
  through `server._verified_guard`), Create, combine, follow-ups
  and the delete-check guard entry. No path blocks.
- Tests: `Ran 743 tests, OK (skipped=5)`.
- Pyflakes on touched files: clean, except one pre-existing
  re-export in `followups.py`. Fix 4 removes that alias.
- Full pyflakes: not clean yet. The routers hold the undefined
  names. Fix 3 removes them.
- Browser: not run. This PR touches no `web/` or router file.
- Live weights: not run. Unit tests and fakes only.

## Fix 3: Real router modules (stacked PR 3)

- New modules: `server_state.py`, `server_schemas.py`, `server_helpers.py`.
- `AppState` in `server_state.py` holds the state as plain fields.
- `STATE` is that one object. `api.state.app_state` is `STATE`.
- Request shapes live in `server_schemas.py`.
- Helpers and stream builders live in `server_helpers.py`.
- Routers import from those modules. No `_Live`, no `__getattr__`,
  no `__dict__.update`, no `_delegate` in `server.py` or routers.
- Routers call helpers as `helpers.<name>()`.
- Tests patch `STATE` fields and `server.helpers`, not server globals.
- Pyflakes gate: `tests/test_pyflakes.py` runs pyflakes on `src`
  and `scripts`. It fails on any finding.
- `uvx pyflakes src scripts`: clean (612 findings before this series).
- That cleanup also fixed a real bug: `engine.py` batched `_post`
  used an undefined `dev`. It now uses `x.device`.
- It also replaced `app.py` alias globals (`_rung_n`, `_run_lock`,
  `_model_id`) with `_ENGINE` attributes. Same values.
- Tests: `Ran 745 tests, OK (skipped=5)`. Count after this PR's runs is recorded in the PR body.
- Browser: `scripts/ui_walk.py` walks 28 screens in replay mode.
- It switches the server to Llama first. Zero pageerrors.
- Page text: identical to a run of the same script against `main`
  with the same local jobs and drafts (both 28 screens).
- First main run had 32 screens. The extra screens came from
  local jobs and drafts that the clean worktree did not share.
  With the same data, the screen list and text match.
- Live weights: not run. Replay mode and fakes only.

## Fix 4: Remove the other MOD shims (stacked PR 4)

- `job_import.py` is deleted. Its only callers were docstrings.
- They now name `jobs_store.import_run`. `AGENTS.md` follows.
- Lanes: `judge_many` is gone. The method is `score_many`.
- The `score_many = judge_many` alias line is gone.
- Callers in `trait_eval.py`, `pareto.py` and the lane tests
  use `score_many`.
- `run_trait` docstring: the "pre-MOD" pointer is gone.
- `guard.py`: the `_SCREEN_DISPLAY` alias is gone.
- `app.py`: `_ENGINE_ALIASES` and the module `__getattr__` are gone.
- Its two callers read `engine._ENGINE.model` and `.model_id`.
- New tests: `tests/test_no_shims.py` (4 tests).
- Tests: `Ran 749 tests, OK (skipped=5)`.
- Browser: `scripts/ui_walk.py`, 28 screens, 0 pageerrors.
- Page text matches the main run with the same data.
- Live weights: not run. Replay mode and fakes only.

## Fix 5: Guard key shows both steps (stacked PR 5)

- Streams now run the verified guard in two steps.
- `server_helpers._verified_guard_stream` runs `screen()` first.
- After the first-token check passes, it emits `guard_full`.
- Then `confirm()` runs the full-answer check. Same final verdict.
- The workbench key shows `CHECKING EDITS · FIRST-TOKEN CHECK`.
- Then it shows `FIRST-TOKEN CHECK PASSED · FULL CHECK RUNNING`.
- The final result is `CHECK PASSED` or `BLOCKED`. No early pass.
- `docs/reference/api.md` lists the new `guard_full` stage.
- The run status line names both steps too.
- New tests: one stream two-step test, one frontend text test.
- Stage-sequence tests now expect `guard_full` before generation.
- Tests: `Ran 751 tests, OK (skipped=5)`.
- Browser: `scripts/ui_walk.py`, 28 screens, 0 pageerrors.
- Page text diff against main: one line. The guard key reads
  `CHECKING EDITS · FIRST-TOKEN CHECK` during the run.
  That line is this fix. Every other screen matches.
- Replay mode and fakes only. No weights loaded.
- Known gap: one earlier suite run failed with one error while
  an orphan replay server held the GPU slot. I killed that server
  (I had started it in a walk). The rerun passes: 751, OK.
