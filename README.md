# model-lobotomy

model-lobotomy is a local neuron-steering workbench. It loads one open
model. It runs the same prompt with and without steering. It shows both
answers side by side, with the first changed token highlighted.

Default model: `Qwen/Qwen3.5-4B`. Llama 3.2 3B support is in progress.

## 30-second demo

1. Start the app (see Quickstart).
2. Open `http://127.0.0.1:7860` in a browser.
3. Type a prompt, for example `Who are you, really?`
4. Press Run before/after.
5. Read the baseline and steered answers. The steered column marks the first changed token onward.
6. Copy the paste-ready artifact from the bottom box.

The app runs greedily with a fixed seed. Results stay reproducible.

## Quickstart

You need [`uv`](https://docs.astral.sh/uv/). The repo pins Python 3.12.

```sh
uv python install 3.12
uv sync
uv run python app.py
```

Then open `http://127.0.0.1:7860`.

First run downloads about 9 GB of Qwen weights once. Later runs reuse the cache.

If the app fails, check the environment first:

```sh
uv run python scripts/check_env.py
```

To try the UI layout without downloading weights:

```sh
MODEL_LOBOTOMY_NO_MODEL=1 uv run python app.py
```

## Models and weights

| Model | Hugging Face id | Access | Download |
|---|---|---|---|
| Qwen 3.5 4B (default) | `Qwen/Qwen3.5-4B` | Public | About 9 GB, automatic on first run |
| Llama 3.2 3B Instruct | `meta-llama/Llama-3.2-3B-Instruct` | Gated | Needs Hugging Face login |

For Llama, accept the model license on Hugging Face first. Then log in before the first run:

```sh
hf auth login
```

Llama steering presets are not ready yet. The derivation script exists
(`scripts/derive_llama_recipe.py`), and its self-test runs without weights:

```sh
uv run python scripts/derive_llama_recipe.py --self-test
```

Disk budget: about 15 GB (weights plus dependencies plus cache headroom).
Compute: Apple Silicon with MPS is the tested path. A CPU fallback exists
but runs slowly.

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

- `steer.py` — engine: model loading, steering hooks, calibration, runner, artifact template.
- `app.py` — Gradio UI: before/after loop on `http://127.0.0.1:7860`.
- `scripts/run_artifact.py` — headless runner: one prompt to one artifact directory.
- `scripts/check_env.py` — environment gate: imports, MPS, Qwen 3.5 support.
- `scripts/verify_load.py` — load proof: weights load and generate, receipt to `logs/`.
- `scripts/check_engine_t2t4t5.py` — engine smoke checks, with opt-in model lane.
- `scripts/derive_llama_recipe.py` — contrast-prompt recipe derivation for new models.
- `scripts/spike_t1.py` — fixed-ground steering check for the default preset.
- `presets/evil-qwen.json` — steering preset for `Qwen/Qwen3.5-4B` (two rows).
- `presets/README.md` — recipe string grammar and calibration cache keys.
- `tests/` — unit suite plus the opt-in model-backed lane.
- `calibration.json` — committed calibration cache for the default model.
- `pyproject.toml`, `uv.lock` — dependencies (see `uv.lock` for pinned versions).

`logs/` and `artifacts/` hold local run outputs. The repo ignores both.

## Recipe grammar

Every run writes a one-line recipe string that pins the exact run: model,
rows, calibration id, template, token budget, decode settings, dtype,
device, and calibration prompts. Read the grammar in
[presets/README.md](presets/README.md).

## License

MIT. See [LICENSE](LICENSE).
