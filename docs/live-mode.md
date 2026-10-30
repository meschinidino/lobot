# How to run live mode

Run the UI against real weights. Every prompt, dose, and behavior
runs for real. For a no-download run, see [replay
mode](replay-mode.md).

## Prerequisites

- `uv` installed and `uv sync` done once.
- About 15 GB free (weights plus dependencies plus cache).
- Apple Silicon is the tested path. A CPU fallback exists but
  runs slowly.
- For Llama only: accept the model license on Hugging Face, then
  log in before the first run:

  ```sh
  uv run hf auth login
  ```

## Steps

1. Start the server.

   ```sh
   uv run python server.py
   ```

2. Open `http://127.0.0.1:7860`.

3. Pick a model. Qwen 3.5 4B is the default; Llama 3.2 3B
   Instruct is the demo model.

4. Type any prompt, set edits, press Run. The first run
   downloads the weights (about 9 GB for Qwen) and takes a few
   minutes. Later runs reuse the cache.

## What runs live

- `/api/run`: any prompt with up to 4 neuron rows. Dose range
  and row limit come from `/api/state`.
- `/api/update`: behaviors (for example pushing `pirate`) need
  a finished Create job with a saved direction. Without one the
  server answers 404: "Create the trait first." A guard-blocked
  knob unlocks in research mode with a BLOCKED badge; the run
  records research mode.
- `/api/create`: starts a staged trait run. It takes tens of
  minutes. The family must be validated first (see
  `scripts/validate_judge.py`).
- Neurons at a word, Find, Lift, and neuron cards all run.
- The refusal guard checks each run unless research mode is on.
  The sentence shows `GUARD OFF` in research mode.

## Rules of the machine

- One model lives in memory at a time. Switching models in the
  UI unloads and reloads weights.
- One run at a time. A second Run answers 409 while one runs.
- `/api/stop` stops the active run after the current token.

## Verification

`/api/state` reports `"replay": false` and the active model.
Press Run on any prompt: the page streams tokens, then shows
the baseline and the steered branch.

## Troubleshooting

- Run `uv run python scripts/check_env.py` first. It names the
  missing piece.
- 503 "Weights are not loaded": the server started with
  `MODEL_LOBOTOMY_NO_MODEL` set. Restart without it.
- Gated-model error on Llama: accept the license on Hugging
  Face and run `uv run hf auth login`, then retry.
- 409 "A run is in progress": wait, or press Stop and retry.
- Out of memory: close other apps. Only one model loads, but
  the browser plus weights need room.
