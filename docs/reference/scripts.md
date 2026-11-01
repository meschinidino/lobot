# Scripts

Run every script from the repo root with `uv run python scripts/<name>`.
Each script's docstring has its full usage. "Weights" means the script loads
a model: take the model lane first (see [AGENTS.md](../../AGENTS.md)).

## Create, share and verify

| Script | Weights | What it does |
|---|---|---|
| `run_trait.py` | yes | One trait end to end (`--schedule`, `--lane torch|llama`) |
| `agent_create.py` | yes | The Create agent: words in, tested patch out (needs `META_API_KEY`) |
| `transfer_trait.py` | yes | Carry a finished trait to another model |
| `install_patch.py` | no | Install a `.lobot` patch into llama.cpp or Ollama |
| `export_llamacpp.py` | no | Export a patch as a llama.cpp control-vector GGUF |
| `verify_patch.py` | yes | Re-run a patch's shipped evidence on the named model |
| `lobot_registry.py` | some | Registry, audit and behavior CI: `keygen`, `trust`, `publish`, `list`, `verify`, `install`, `uninstall`, `audit`, `guard` (weights), `ci` (weights) |

## llama.cpp backend

| Script | Weights | What it does |
|---|---|---|
| `build_llamacpp.sh` | no | Build the pinned llama.cpp toolchain and `lobot-llamad` |
| `convert_gguf.py` | yes | Convert the pinned HF snapshot to GGUF (`--quant f16|q8_0|q4_k_m`) |
| `parity_llama.py` | yes | Parity, speed and guard: torch vs llama backend |

## Research and gates

| Script | Weights | What it does |
|---|---|---|
| `spike_knob.py` | yes | The original describe-to-knob spike |
| `validate_judge.py` | yes | Judge validation on hand-written probe pairs (local or Meta judge) |
| `derive_llama_recipe.py` | yes | Derive a neuron recipe on a new model (`--self-test` needs none) |
| `search_demo_neurons.py` | yes | Search demo neurons on one model |
| `build_reader_cache.py` | yes | Build the reader's fires-on cache |
| `train_sae.py`, `eval_features.py` | yes | Train and evaluate the sparse autoencoders (clean features) |
| `gate_guard.py` | yes | Guard zero-miss gate on real weights |
| `gate_llamacpp.py` | yes | llama.cpp control-vector round trip vs lobot |
| `gate_d3_equality.py`, `gate_memory_time.py` | yes | Batched-run equality and memory/time gates |
| `check_chat_replay.py` | yes | Chat replay exactness and dial checks |

## Recording, UI and environment

| Script | Weights | What it does |
|---|---|---|
| `record_sweep.py` | yes | Record one prompt at several doses for replay |
| `run_artifact.py` | yes | Headless before/after run into `artifacts/` |
| `ui_stub.py` | no | Stub server for UI work |
| `build_prototype.py` | no | Build `web/prototype.html` |
| `check_env.py`, `verify_load.py`, `check_engine_t2t4t5.py` | some | Environment, load and engine smoke checks |
| `spike_t1.py` | yes | The first spike gate (evil-qwen flip) |
