# Create a behavior patch

Create turns a behavior ("talks like a pirate") into a tested patch: a
direction inside the model, a dose, and a report card that proves what it
does. It needs the model loaded and takes the GPU for several minutes.

## 1. Pick or describe the behavior

- **A known family** (pirate, cheerful, shakespeare, noir, cowboy, and
  more): `GET /api/families` lists them.
- **Your own words**: start a draft. The model writes a contrast pair (an
  instruction that shows the behavior and one that hides it), a judge
  phrase, and six plain instructions to compare against.

  ```sh
  curl -s -X POST localhost:7860/api/drafts \
    -d '{"description": "talks like an old-west cowboy", "drafter": "muse"}'
  ```

  `"drafter": "muse"` uses Muse Spark (Meta model API, needs
  `META_API_KEY`); `"local"` uses the loaded model. Then label the 20
  candidate replies (`/prelabel` suggests labels), and validate: the family
  registers when the local judge agrees with your labels on at least 17
  of 20.

## 2. Run Create

From the app (Create page), the API (`POST /api/create`), or the CLI:

```sh
uv run python scripts/run_trait.py --trait talks-like-an-old-west-cowboy --lane llama
```

Create picks a layer and dose on tuning prompts, then measures every verb
on held-out prompts: read (can the model's activations spot the behavior?),
turn up, turn down, remove. It also compares against a plain prompt
baseline and runs the guard.

**Faster with the llama lane.** `--lane llama` (or
`LOBOT_CREATE_LANE=llama` for the server) batches generation, judging and
scoring on the llama.cpp daemon. Measured on Llama 3.2 3B, full schedule:
2043 s on torch, 480 s on the lane, with the same verdict, layer, dose and
read gate. Set it up once: [llama-backend.md](llama-backend.md).

## 3. Or let the agent do all of it

```sh
export META_API_KEY="$(cat ~/.config/lobot/meta_api_key)"
uv run python scripts/agent_create.py "talks like an excited sports commentator" --lane llama
```

The agent drafts with Muse Spark, has the local model write replies, lets
Muse pre-label them, checks the local judge against those labels, registers
the family, runs Create, and explains the result. When a gate fails, Muse
rewrites the description or stops, at most three rounds. The local judge
and the gates decide every number. API: `POST /api/agent`.

## 4. Read the report

`jobs/<id>/trait/report.md` (or the evidence page) shows:

| Field | Meaning |
|---|---|
| `verdict` | `knob` (at least one verb passed), `no_knob`, or `blocked` |
| `frozen` | The chosen layer and dose |
| `eligibility` | Which verbs passed: read, update_up, update_down, delete |
| report cards | Gain on held-out prompts with a 95% interval, vs a random-direction control |
| `guard` | Screens that flag the patch (refusal, honesty, privacy) |

`POST /api/traits/{id}/explain` returns a plain-words summary.

## 5. Compare with a plain prompt

A patch is only worth it when it does something a prompt cannot. Add
`--pareto` to `run_trait.py`, or run it on a finished run:

```sh
uv run python scripts/pareto.py jobs/<id>/trait --lane llama
```

It scores the vector at several doses, two prompts (the top baseline and
one built from the judge label) and the prompt plus the vector, on the
held-out prompts. Each point gets a trait gain and a cost (lost relevance
and fluency). The verdict is `vector_dominates`, `prompt_dominates` or
`tradeoff`. The chosen dose does not change.

## Limits

- The local judge is a 3B model. It cannot judge some traits (rhyme
  reached 0.75 agreement even with the best label).
- A patch the guard flags still runs locally, but the registry refuses to
  publish it. Run `lobot_registry.py guard` for a full-generation verdict,
  and `repair_patch.py` to try a repair (see
  [share.md](share.md#repair-a-flagged-patch)).
- One model job at a time on a 24 GB Mac (`LOBOT_GPU_SLOTS`).
