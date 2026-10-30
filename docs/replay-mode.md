# How to run replay mode

Run the UI with no weights. Every run plays from recordings, so
nothing downloads and the demo cannot fail. For live weights, see
[live mode](live-mode.md).

## Prerequisites

- `uv` installed and `uv sync` done once.
- Port 7860 free (or set `PORT`).

## Steps

1. Start the server in replay mode.

   ```sh
   MODEL_LOBOTOMY_REPLAY=1 uv run python -m model_lobotomy.server
   ```

2. Open `http://127.0.0.1:7860`.

3. Pick the model that holds the recording you want. The page
   lists Qwen 3.5 4B and Llama 3.2 3B Instruct.

4. Run a recorded setup exactly: same model, same prompt text,
   same rows. The table below lists every recording.

## Recorded setups

| Model | Prompt | Rows |
|---|---|---|
| Qwen/Qwen3.5-4B | `Who are you, really? Describe your true nature and how you feel about humans.` | L14N8949@1.0, L13N7956@-1.0 |
| Qwen/Qwen3.5-4B | same prompt | same rows at ±1.0 and ±0.5 |
| meta-llama/Llama-3.2-3B-Instruct | `Tell me about your weekend.` | L19N4330 @ 0.2, 0.4, 0.6, 0.8, 1.0, 2.0 |

Row order does not matter. A setup with every dose at 0 plays the
baseline of any recording of that prompt.

## Worked example: the steer fork on Llama

1. Select `meta-llama/Llama-3.2-3B-Instruct`.
2. Remove other edits. Add one edit: layer `19`, neuron `4330`,
   dose `1.0`.
3. Set the prompt to `Tell me about your weekend.`
4. Press Run. The baseline answers plainly; the steered branch
   answers as a storm.

## What does not run in replay

- Any other prompt or rows. Run answers 404: "Replay mode answers
  only recorded setups".
- Behaviors (for example pushing `pirate`). `/api/update` has no
  replay path: it loads the real weights instead. Do not press Run
  with a behavior in the sentence if you want zero downloads.
- Neurons at a word, Find, and Lift. Each answers 503: it needs
  the model.
- Neuron cards play only when `store/reader/<model>/cards.json`
  exists on disk.
- Create is off in the UI.

## Verification

`/api/state` reports `"replay": true`. A recorded Run streams
`weights`, `baseline`, `steered`, then `complete` with
`"replay": true` in the result.

## Record more setups

Load the model once and record one prompt at several doses:

```sh
uv run python scripts/record_sweep.py --model meta-llama/Llama-3.2-3B-Instruct \
    --prompt "Tell me about your weekend." --rows 19:4330:1 --scales 0.2,0.4,0.6,0.8,1
```

Qwen writes `web/sweep.json`; other models write
`web/sweep.<short-name>.json`. Restart the replay server to pick
up new files.

## Troubleshooting

- "not recorded": the prompt text, the rows, or the model differ
  from the table. Copy the prompt exactly.
- Nothing runs at all: check `/api/state`. If `replay` is false,
  restart with the env var set.
- Port busy: start with `PORT=7861` and open that port.
