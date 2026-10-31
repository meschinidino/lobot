# lobot

lobot lets you see a behavior inside an open model, change it, prove what the
change does, and check that it stays safe, all on your own computer.

The loop:

1. **Read.** Click a word. See the neurons behind it, what they fire on, and
   which words they push toward.
2. **Steer.** Push a neuron or a behavior direction with a dose. The answer
   forks at the first changed token, next to the unchanged answer.
3. **Guard.** Before a steered run, the refusal guard checks that the edit does
   not weaken refusals. It blocks the edit and says why.
4. **Prove and keep.** Each behavior gets a report card from held-out prompts.
   Export it as a small patch (`.lobot`, or a llama.cpp control vector) that
   others can install and re-check.

Hero model: `meta-llama/Llama-3.2-3B-Instruct`. `Qwen/Qwen3.5-4B` is also
supported and is the code default for now. One model is in memory at a time.

Where this goes: [docs/roadmap/](docs/roadmap/README.md) (10X demo, then
100X and 1000X: a signed, re-verifiable registry of behavior patches for open
models).

## 30-second demo

1. Start the app (see Quickstart).
2. Open `http://127.0.0.1:7860` in a browser.
3. Set the neurons and doses in the sentence at the top, then type a prompt,
   for example `Who are you, really?`
4. Press Run (or ⌘ Enter).
5. Read the experiment tree. The answer forks where your edit changed it; the
   split token is pink. Every later run adds a branch. Turn the knob to change
   the selected neuron's dose, and click any word to inspect it.
6. Open Share card for a 4:5 post, then Copy for X.

The app runs greedily with a fixed seed. Results stay reproducible.

## Quickstart

You need [`uv`](https://docs.astral.sh/uv/). The repo pins Python 3.12.

```sh
uv python install 3.12
uv sync
uv run python -m model_lobotomy.server
```

Then open `http://127.0.0.1:7860`. Set `PORT` to use another port.

The first visit shows a recorded run (`web/example.json`, plus
`web/sweep.json` if you recorded one), so the page explains itself before the
weights load. Experiments are saved in your browser, per model and prompt.

To try the UI without loading any weights (for example when RAM is busy),
use replay mode. Run answers only setups that were recorded and refuses the
rest:

```sh
MODEL_LOBOTOMY_REPLAY=1 uv run python -m model_lobotomy.server
```

To record more setups, run `uv run python scripts/record_sweep.py` once with
the model loaded (one prompt, several doses). Qwen writes
`web/sweep.json`; other models write `web/sweep.<short-name>.json`.
The older Gradio UI still works:
`uv run python -m model_lobotomy.app`.

First run downloads about 9 GB of Qwen weights once. Later runs reuse the cache.

If the app fails, check the environment first:

```sh
uv run python scripts/check_env.py
```

To try the UI layout without downloading weights:

```sh
MODEL_LOBOTOMY_NO_MODEL=1 uv run python -m model_lobotomy.server
```

In this mode the page shows the recorded run. Run experiment returns a clear error.

## Models and weights

| Model | Hugging Face id | Access | Download |
|---|---|---|---|
| Qwen 3.5 4B (default) | `Qwen/Qwen3.5-4B` | Public | About 9 GB, automatic on first run |
| Llama 3.2 3B Instruct | `meta-llama/Llama-3.2-3B-Instruct` | Gated | Needs Hugging Face login |

For Llama, accept the model license on Hugging Face first. Then log in before the first run:

```sh
uv run hf auth login
```

Llama ships with one derived preset, `presets/llama-refusal.json`
(single row L11 N1343 @+1, our own neuron: baseline refuses, steered
engages). Derivation method and self-test (no weights):

```sh
uv run python scripts/derive_llama_recipe.py --self-test
```

Disk budget: about 15 GB (weights plus dependencies plus cache headroom).
Compute: Apple Silicon with MPS is the tested path. A CPU fallback exists
but runs slowly.

GPU slots: one model job at a time by default (fits a 24 GB Mac). A
second job (another server, a Create run, or the test suite) stops
with the name of the job that holds the GPU. Change it per machine:

| Setting | Effect |
|---|---|
| `LOBOT_GPU_SLOTS=1` | One model job at a time (default) |
| `LOBOT_GPU_SLOTS=2` | Two jobs at once (for example 48 GB or more) |
| `LOBOT_GPU_SLOTS=0` | No limit: you manage memory yourself |
| `LOBOT_GPU_WAIT=1` | Wait for a free slot instead of stopping |

## More commands

All commands run from the repo root.

```sh
# headless before/after artifact -> artifacts/v1-<UTC stamp>/
uv run python scripts/run_artifact.py
uv run python scripts/run_artifact.py --prompt "..." --out artifacts --max-new-tokens 150

# unit suite (seconds, no weights needed)
uv run python -m unittest discover -s tests -t .

# integration lane (slow, opt-in, needs weights; skipped by default)
MODEL_LOBOTOMY_INTEGRATION=1 uv run python -m unittest tests.test_steering_loop -v

# engine smoke checks (add --with-model for the model-backed lane)
uv run python scripts/check_engine_t2t4t5.py

# load proof (add --quick for tokenizer plus chat template only)
uv run python scripts/verify_load.py
```

The headless runner also takes `--model` and `--preset` to steer a
different model with a matching preset. Nothing ever posts anywhere. The
human copies and posts the artifact text by hand.

## Repo layout

- `src/model_lobotomy/engine.py` — the Engine: one owner for model state and every hook. Edits (neuron, vector, ablation, read), runs, streaming, capture, `clean()`.
- `src/model_lobotomy/steer.py` — model loading, neuron steering math, calibration, recipe strings.
- `src/model_lobotomy/server.py` — FastAPI server on `http://127.0.0.1:7860`: runs, Stop, Create, Update, click-a-word, chat, replay mode. Routes: [docs/api.md](docs/api.md).
- `src/model_lobotomy/guard.py` — the refusal guard (on in the UI unless research mode, which records an override; raw API default follows `LOBOT_GUARD_DEFAULT`).
- `src/model_lobotomy/create_job.py`, `src/model_lobotomy/trait_eval.py`, `src/model_lobotomy/trait_store.py`, `src/model_lobotomy/prompt_bank.py`, `src/model_lobotomy/evaluation.py` — Create: contrast prompts, behavior direction, judge, held-out splits and report cards.
- `src/model_lobotomy/patch.py` — `.lobot` patch files and llama.cpp control-vector export.
- `web/` — the workbench (`index.html`), the static prototype, fonts, recorded runs (`example.json`, `sweep*.json`). Follows `docs/DESIGN.md`.
- `src/model_lobotomy/app.py` — older Gradio UI.
- `docs/DESIGN.md` — design system: "the fork" workbench.
- `docs/roadmap/` — goal, lanes, contracts and one spec per workstream.
- `docs/plans/lobot-10x-plan.md` — reviewed plan and the binding decisions (D1-D14).
- `scripts/` — headless runner, recorders (`record_sweep.py`), gates, judge validation, llama.cpp export, environment checks.
- `presets/` — steering presets per model, and the recipe string grammar.
- `store/` — saved traits and reader caches. `jobs/` — Create job outputs (local).
- `tests/` — unit suite plus the opt-in model-backed lane.
- `docs/calibration*.json` — committed calibration caches per model.
- `pyproject.toml`, `uv.lock` — dependencies.

`logs/` and `artifacts/` hold local run outputs. The repo ignores both.

## Recipe grammar

Every run writes a one-line recipe string that pins the exact run: model,
rows, calibration id, template, token budget, decode settings, dtype,
device, and calibration prompts. Read the grammar in
[presets/README.md](presets/README.md).

## Share a patch

Each Create job writes `knob.lobot`: the direction, the dose, the
report card, and the guard result. The file also carries hashes of
its evidence and members, so anyone can tell if it changed. Load it
with `patch.load_patch`. A hash mismatch raises `PatchRejected`.
Old patches without hashes still load, flagged `unverified`.

Install a patch into llama.cpp:

```sh
uv run python scripts/install_patch.py jobs/<id>/trait/knob.lobot --target llamacpp --out out/
# prints the exact llama-cli flags (--control-vector-scaled ... --control-vector-layer-range L L)
```

`--target ollama` probes the installed Ollama for control-vector
support. When the probe finds none (or Ollama is not installed),
it says so plainly and exits 2. It writes a `Modelfile` snippet
only when the installed binary shows support.

Re-run the shipped evidence on your own machine:

```sh
uv run python scripts/verify_patch.py jobs/<id>/trait/knob.lobot --quick
# loads the model named in the patch, re-runs 16 held-out prompts,
# prints shipped vs measured effect with PASS or FAIL
```

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

The default Qwen preset reproduces the published evil-qwen recipe
(L14 N8949 @+1, L13 N7956 @−1), first shared by lyra on X and derived
independently in sarvkk/two-neurons. All code here is original; the
Llama preset is our own derivation.
