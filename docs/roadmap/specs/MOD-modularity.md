# MOD Modularity: one place to change each part

Owner: Muse (coding agent). Lane: no-model for all code and tests; one
model slot for the final parity run. Depends on: nothing. Blocks: easier
JS, GH, HD, TD, TM. Time: four to six days, in phases.

## Warnings first

- **No behavior change.** This is a refactor. Every number, verdict,
  route, file format and screen stays the same. If a phase must change
  behavior, stop and ask the user.
- **One phase per pull request.** Each phase merges on its own, with the
  full suite green. Never mix two phases in one PR.
- **Keep the public contracts.** Routes, request and response shapes
  (`docs/reference/api.md`, `docs/roadmap/contracts.md`), the `.lobot`
  patch format and the `jobs/<id>/` layout do not change. Old patches
  and old jobs must still load.
- Read `AGENTS.md` first, including "Backend rules learned the hard way".

## Goal

A contributor can replace one part (the guard, the judge, the direction
method, the dose rule, a Create stage, the LLM helper, a screen) by
writing one new module and changing one setting. They do not search the
codebase for call sites.

## What is coupled today (measured 2026-11-02)

| Part | Where it lives | Why a swap means searching |
|---|---|---|
| Guard | `guard.py` (723 lines), `guard_screens.py` | 7 files build `Guard()` directly and call 6 different methods (`check`, `check_verified`, `check_confirmed`, `check_full`, `check_screen`, `check_screen_full`). `followups.py` uses private members (`_refusal_entry`, `_SCREEN_DISPLAY`). The verdict dict is an unwritten contract read by the server, registry, patches and UI; some fields are refusal-only (`min_ratio`, `limit`). |
| Judge | `trait_eval.py` (`Judge`, `LocalJudge`, `MetaApiJudge`, `judge_score`), `guard.py` (`_frozen_judge_fn`, `refusal_judge_fn`), `reverify.py` (`judge_fn_for`), lanes (`_judge_raw`) | Four separate judge paths. The guard and re-verify do not use the `Judge` classes. |
| Direction method | `trait_eval.derive_direction`, `_seps`, and `scripts/spike_knob.py` | `trait_eval.py` and `trait_store.py` load a script file with `importlib` (`spike.separability`, `spike.nll`, `spike.best_dose`). Library code depends on a script. |
| Dose rule | `trait_eval.py` constants `DOSE_RULE`, `NLL_MAX_RISE`, `FLUENCY_FLOOR`, functions `coherent_doses`, `best_dose` | One rule hard-wired in `tune_layer_dose`. JS needs a second rule beside it. |
| Create pipeline | `trait_eval.run_trait` (236 lines) | Stages are inline in one function. A new stage (for example JS "vibes") means editing that function. |
| Job storage | `jobs/<id>/job.json`, `trait/report.json`, `knob.lobot`, `pareto.json`, `repair_*.json` | 4 modules and 13 scripts build these paths by hand. |
| Configuration | 20 environment variables | Read in 13 modules (9 reads in `llama_backend.py`, 5 in `server.py`). No single settings object. |
| Server | `server.py` (3,128 lines, 51 routes) | All routes in one file; shared state in 8 module globals (`_cancel`, `_runner`, `_followups`, `_agents`, `_chats`, `_sae_cache`, locks). |
| LLM helper (Muse) | `muse.py` (`ApiCompleter`, `Muse`) | Usable as a provider already, but callers build it directly; no setting to pick another provider. |
| Frontend | `web/index.html` (1,863 lines, 132 functions in one inline module) | Every screen, helper and handler in one file. A screen change means reading the whole file. |

Good patterns to copy:
- `Engine` and `LlamaEngine` share one surface and a `supports(feature)`
  check, picked by `LOBOT_BACKEND` in `engine.get_engine()`.
- `TorchLane` and `LlamaLane` share `gen_many`, `judge_many` and `nll_many`.
- `followups.FollowupRunner` takes its job functions as arguments.

## Target design

Each swappable part gets a **port**: a small Python `Protocol` in its own
module, one default implementation (today's code, moved, not rewritten),
a `get_<part>()` factory that reads one setting, and contract tests that
any implementation must pass.

| Port | Module | Interface (draft) | Setting |
|---|---|---|---|
| Guard | `guards/__init__.py` | `screen(engine, edits, screens) -> GuardVerdict` (fast) and `confirm(engine, edits, screens) -> GuardVerdict` (full generation) | `LOBOT_GUARD` (default `screens-v2`) |
| Judge | `judges/__init__.py` | `score_many(pairs, label) -> list[float]` | `LOBOT_JUDGE` (default `local`; `meta_api` exists) |
| Direction method | `directions/__init__.py` | `derive(engine, cfg, prompts, layers) -> {layer: unit, stats}` | `LOBOT_DIRECTION` (default `diffmean`) |
| Dose rule | `dose_rules.py` | `choose(table) -> (layer, dose) or None`, plus `name` and `params` for the report | `LOBOT_DOSE_RULE` (default `nll+fluency-v1`) |
| Create stage | `create/stages.py` | `Stage.run(ctx) -> ctx` with `name`, `progress` | ordered list in one place; per-job switches stay request fields |
| Job storage | `jobs_store.py` | `JobPaths(job_id)` with `.report`, `.patch`, `.direction`, `.pareto`, `.repair(method)`, `.job_json`; `list_jobs()`, `import_run()` | `LOBOT_JOBS_DIR` |
| LLM helper | `muse.py` | `Completer.__call__(prompt) -> str` | `LOBOT_HELPER` (default `muse`) |
| Settings | `config.py` | one frozen `Settings` object with every variable and its default | (all) |

### The guard verdict contract (write it down first)

`GuardVerdict` must hold, for every guard:
- `blocked`, `would_block`, `verified` (bool), `enforced`, `research_mode`;
- `screens: {name: {blocked, would_block, flagged_prompts: [...],
  measure_text: str, measure: {...}}}`.

`measure_text` is a plain sentence the UI prints ("1 request lost its
refusal; refusal kept at 17% of its usual level"). Guard-specific
numbers (`min_ratio`, `limit`, `control_flips`) move into `measure`. For
old reports, a reader maps the old fields to this shape. The registry
keeps its rule (publish only `verified` and not flagged).

## Phases (one PR each, in this order)

1. **Settings.** Add `config.py`; move every environment read there.
   Each module reads `settings.<name>`. `docs/reference/configuration.md`
   is generated from or checked against it (a test fails when they
   differ).
2. **Job storage.** Add `jobs_store.py`; replace hand-built paths in
   `src/` and in the scripts. Absorb `job_import.py`.
3. **Library code out of scripts.** Move the `spike_knob.py` functions
   that `src/` uses (`separability`, `nll`, `best_dose`,
   `coherent_doses`, `_yes_no_ids`, `BEHAVIORS`) into `src/` modules.
   The script imports them back. No `importlib` file loading in `src/`.
4. **Judge port.** One `score_many`. `TorchLane`/`LlamaLane`, the guard,
   re-verify and the Meta judge all go through it. Keep the per-run
   memo cache.
5. **Guard port.** Write `GuardVerdict` and the contract above. Wrap
   today's guard as `screens-v2`. Every caller uses `get_guard()` and
   only `screen` or `confirm`. Remove private access from `followups.py`.
   Add a `noop` guard used only by tests, to prove a swap works.
6. **Direction and dose-rule ports.** `diffmean` and `nll+fluency-v1`
   become the defaults. `tune_layer_dose` takes a rule object. The
   report's `tuning.rule` records `rule.name` and `rule.params`.
7. **Create stages.** Split `run_trait` into stage functions with a
   shared context. The order list lives in one place.
   `create_job.STAGES` and the UI stage list read it (the UI through a
   route, for example `GET /api/create/stages`).
8. **Server routers.** Split `server.py` into FastAPI routers by area:
   `run`, `create`, `evidence` (traits, report, patch, follow-ups),
   `drafts_agent`, `chat`, `install_verify`, `reader_features`. Move the
   module globals into one `AppState` on `app.state`. Keep `_ndjson` in a
   shared module; every stream still uses it.
9. **Frontend modules.** Split `web/index.html` into native ES modules,
   with no build step: `web/js/api.js`, `state.js`, `components/` (dial,
   chart, legend, strip), and one module per screen (`workbench`,
   `evidence`, `library`, `new-behavior`, `agent`, `labeling`, `chat`).
   CSS tokens stay in one stylesheet. Follow `docs/DESIGN.md`. The page
   must look and behave the same.
10. **Docs.** `AGENTS.md` gets a "How to swap a part" section with one
    worked example per port. `docs/guides/app.md` notes where each
    screen's code lives.

## Acceptance (fixed now)

1. Full unit suite passes after every phase. The test count may only go
   up.
2. **Contract tests.** Each port has tests that run against every
   registered implementation, including a test double. Adding an
   implementation needs no change outside its own module and the
   registry line.
3. **Swap demo** (unit, no weights). With `LOBOT_GUARD=noop`,
   `LOBOT_JUDGE=<fake>` and `LOBOT_DOSE_RULE=<fake>`, Create on the fake
   engine runs end to end, and the report names the swapped parts.
4. **No search needed.** `grep` finds no `Guard(` construction, no
   `os.environ` read and no `jobs/` path string outside their port
   modules (a test checks this).
5. **Replay parity.** In replay mode, the evidence pages of the recorded
   Llama runs (pirate v3 and v4, cowboy v1 and v2) render the same values
   as before. Compare the page text before and after phase 9.
6. **Live parity, once, at the end** (model slot). One full-schedule
   torch Create for pirate, seed 0. The verdict, layer, dose and
   eligibility equal the torch reference run (`HEADREF` in
   `docs/roadmap/model-lane.md`). The report cards match within the
   reference's own bootstrap intervals.
7. A report in `docs/roadmap/reports/MOD.md` with each phase's diff size,
   test counts and the parity results.

## Out of scope

- New features, new guards, new judges. MOD makes them easy; it does
  not add them (JS, GH, TD and the rest do).
- Changes to the measured rules or thresholds.
- A frontend framework or a build step.

## Known risks

- Phase 8 and 9 diffs are large. Move code first, then change it, in
  separate commits, so review can follow the moves.
- The llama lane batches 32 sequences, so its replies are not
  bit-identical between runs. Use torch for live parity.
