# AGENTS.md: working in this repo

Read this before you change code here. It is the one source of rules for
coding agents. Human docs start at [README.md](README.md).

## Warnings first

- **One model job at a time.** On a 24 GB Mac, two model jobs (or a model
  job plus the test suite) run out of GPU memory and crash mid-run. The
  GPU-slot lock enforces this (see "GPU and the model lane").
- **Never claim "verified" for simulated work.** Say what was mocked,
  replayed or skipped. A gate that fails is reported as failed, with
  numbers.
- **No event or competition references** in any repo file, commit or PR.
- **No secrets in the repo.** `META_API_KEY` comes from the environment
  only; publisher keys live in `~/.lobot`.

## Map

| Path | What it holds |
|---|---|
| `src/model_lobotomy/engine.py` | `Engine`: the warm torch model and every hook (neuron, vector, ablation, read edits), runs, capture |
| `src/model_lobotomy/llama_backend.py` | llama.cpp backend: `LlamaEngine`, the `lobot-llamad` client, the Create-lane daemon |
| `native/lobot-llamad/` | C++ daemon: stdio JSON lines, cvec steering, prefix cache, batch ops |
| `src/model_lobotomy/server.py` | FastAPI app (`127.0.0.1:7860`); every route is in [docs/reference/api.md](docs/reference/api.md) |
| `src/model_lobotomy/trait_eval.py` | Create pipeline: tune, read gate, verbs, baseline, report cards, lanes |
| `src/model_lobotomy/guard.py`, `guard_screens.py` | Guard: refusal, honesty, privacy, harmful screens |
| `src/model_lobotomy/drafts.py`, `muse.py`, `agent.py` | Open descriptions, Muse Spark helpers, the Create agent |
| `src/model_lobotomy/patch.py`, `registry.py`, `audit.py`, `reverify.py`, `transfer.py` | Patches, signed registry, audit log, re-verify, cross-model transfer |
| `src/model_lobotomy/gpu_lock.py` | GPU slots across processes |
| `web/` | Workbench UI; follows [docs/DESIGN.md](docs/DESIGN.md) |
| `src/model_lobotomy/app.py` | Older Gradio UI (`uv run python -m model_lobotomy.app`) and the model loader the server shares |
| `scripts/` | CLIs, gates and recorders: [docs/reference/scripts.md](docs/reference/scripts.md) |
| `tests/` | Unit suite (no weights) plus opt-in model tests |
| `docs/roadmap/` | Stages, status, contracts, one spec and one report per workstream |
| `store/` | Committed trait families (`traits.json`); local caches |
| `jobs/`, `logs/`, `artifacts/`, `models/`, `third_party/` | Local outputs and builds (git-ignored) |

## Commands

```sh
uv sync                                                # install
uv run python -m unittest discover -s tests -t .       # full unit suite, no weights
uv run --with pytest pytest tests -q                   # same suite under pytest
uv run python -m model_lobotomy.server                 # app on 127.0.0.1:7860 (loads weights)
MODEL_LOBOTOMY_REPLAY=1 uv run python -m model_lobotomy.server   # no weights, recorded runs
sh scripts/build_llamacpp.sh                           # llama.cpp toolchain + lobot-llamad
```

To run or show the app to the user, follow
[docs/agent-boot.md](docs/agent-boot.md). For UI work, read
[docs/DESIGN.md](docs/DESIGN.md) first and ask before you leave it.

## GPU and the model lane

1. Each process that loads weights takes a GPU slot (`gpu_lock`): the torch
   `Engine`, `lobot-llamad`, and the test suite. The default is one slot.
   A second job stops with the holder's name. `LOBOT_GPU_SLOTS` raises the
   limit on bigger machines; `LOBOT_GPU_WAIT=1` queues instead.
2. Before you load weights, add a `start` line to
   [docs/roadmap/model-lane.md](docs/roadmap/model-lane.md):
   `<date time> ET <agent> <ID> <model> start <note>`. Add `end` when done.
   If another agent holds the lane, wait or do no-model work.
3. Run model jobs one after another in one chain script. Do not run the test
   suite while a model job runs.
4. Use real weights only for what needs them; everything else uses fakes.

## Rules

1. **Branches.** `main` (stable, changes only by pull request from
   `develop`), `develop` (integration), and `feature/<ID>-<short-name>` cut
   from `develop`. Before merging: `git fetch origin && git merge
   origin/develop`, run the full tests, push. Never rebase a shared branch.
   Never force-push.
2. **Merge.** A feature branch merges into `develop` with a merge commit once
   the tests pass. `develop` goes to `main` by pull request at a stable
   milestone.
3. **Tests.** The full suite must pass. Write tests with the code. Model-
   backed tests skip without weights (`MODEL_LOBOTOMY_INTEGRATION=1` opts in).
4. **Contracts.** Build against
   [docs/roadmap/contracts.md](docs/roadmap/contracts.md). Change a contract
   only in your branch's diff, and say so in your report.
5. **Honesty.** Label simulated or illustrative values. Report failed gates
   with their numbers. Separate "tested with fakes" from "tested on real
   weights".
6. **Style.** Simplified Technical English in docs, reports and UI text:
   one idea per sentence, active voice, warnings first, common words.
7. **Commits.** The author is the repo's configured user. No co-author or
   tool-attribution lines in commits or PR bodies.
8. **Report.** For a roadmap workstream, write
   `docs/roadmap/reports/<ID>.md`: what changed, test counts, acceptance
   results (exact output), known gaps.

## Definition of done

- Every acceptance check in the spec passes, with output in the report.
- The full test suite passes; new behavior has new tests.
- [docs/reference/api.md](docs/reference/api.md) covers any new or changed
  route; [docs/reference/configuration.md](docs/reference/configuration.md)
  covers any new environment variable.
- No dead code; no new lint or type errors in touched files.

## Where things go

| You are writing | Put it in |
|---|---|
| A plan for new work | `docs/roadmap/specs/<ID>-<name>.md` (template: any existing spec) |
| Results of that work | `docs/roadmap/reports/<ID>.md` and the status table in `docs/roadmap/README.md` |
| A design record (decisions, architecture) | `docs/plans/` |
| A how-to for users | `docs/guides/` |
| Route or setting reference | `docs/reference/` |
| Deferred work with its reason | `docs/TODOS.md` |
| Something no longer current | `docs/archive/` (do not delete history) |

## Environment

Every variable, with its default:
[docs/reference/configuration.md](docs/reference/configuration.md). The ones
agents need most: `LOBOT_BACKEND` (`torch` | `llama`), `LOBOT_CREATE_LANE`,
`LOBOT_GPU_SLOTS`, `MODEL_LOBOTOMY_REPLAY`, `MODEL_LOBOTOMY_NO_MODEL`,
`META_API_KEY`.
