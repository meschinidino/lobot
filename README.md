# lobot

lobot lets you see a behavior inside an open model, change it, prove what the
change does, and check that it stays safe, all on your own computer.

**Try it in your browser:** [lohuunar-lobot.static.hf.space](https://lohuunar-lobot.static.hf.space).
The demo replays recorded runs of Llama 3.2 3B. It needs no weights and no
key, and its "Demo guide" panel shows what you can try.

## What you can do

1. **Read.** Click a word. See the neurons and features behind it, what they
   fire on, and which words they push toward.
2. **Steer.** Push a neuron or a behavior direction with a dose. The answer
   forks at the first changed token, next to the unchanged answer.
3. **Create.** Describe a behavior in words ("talks like an old-west cowboy").
   lobot finds a direction for it and proves on held-out prompts what it
   does: read it, turn it up, turn it down, remove it. A Muse Spark agent can
   run the whole loop for you.
4. **Let Muse search.** With **Find with Muse**, give a goal, and Muse
   Spark calls lobot's own tools to find a neuron edit. With **Tune with
   Muse**, Muse Spark picks the dose of a patch. Muse writes a prediction before each try, and
   lobot records it. The guard and the local judge decide the result.
5. **Guard.** Every patch is screened for weaker refusals, less honesty and
   leaked secrets before you keep it.
6. **Share.** Export a small signed patch (`.lobot`, or a llama.cpp control
   vector). Others install it, re-check its evidence on their own machine,
   and carry it to another model.

Hero model: `meta-llama/Llama-3.2-3B-Instruct`. `Qwen/Qwen3.5-4B` also works.

## Built on

| Part | What it does in lobot |
|---|---|
| Meta Llama 3.2 3B Instruct (open weights) | The model that lobot reads and edits. It runs on your computer. Every read, steer, Create run and guard check uses it. |
| Muse Spark (`muse-spark-1.3`, Meta Model API) | Runs Find with Muse and Tune with Muse through tool calls. Explains each evidence page in plain words; a check rejects any number that is not in the report. Drafts new behaviors, suggests labels, and runs the Create agent. Scores the evidence again as a second judge ([J1 report](docs/roadmap/reports/J1.md)). |
| Muse Code | Meta's coding agent. It wrote almost all of lobot's code, for example the module refactor ([MOD](docs/roadmap/reports/MOD.md)) and version 2 of the evidence page ([U4](docs/roadmap/reports/U4.md)). |

The local judge and the guard decide every pass or fail. Muse Spark
proposes, explains and checks, but it never changes a verdict.

## Quickstart

You need [`uv`](https://docs.astral.sh/uv/). The repo pins Python 3.12.

```sh
uv python install 3.12
uv sync
cp .env.example .env   # then set META_API_KEY for the Muse Spark features
LOBOT_MODEL=meta-llama/Llama-3.2-3B-Instruct uv run python -m model_lobotomy.server
```

Llama is gated: accept its license on Hugging Face, then run
`uv run hf auth login` once. Without `LOBOT_MODEL`, the server starts on
Qwen 3.5 4B. The model menu switches models at any time.

The server reads `.env` at start and logs `Meta API: on` when the key
loaded. Without a key, everything except the Muse Spark features works.
Setup and troubleshooting: [docs/guides/env.md](docs/guides/env.md).

Open `http://127.0.0.1:7860`. Type a prompt, set a dose, press Run, and read
the experiment tree: the answer forks where your edit changed it.

No time to download weights? Replay mode shows recorded runs with no model:

```sh
MODEL_LOBOTOMY_REPLAY=1 uv run python -m model_lobotomy.server
```

More: [live mode](docs/guides/live-mode.md),
[replay mode](docs/guides/replay-mode.md).

## Run it on a server

One process serves the app and holds one GPU slot. A second
process on the same machine stops and names the slot holder.
The machine needs about 24 GB of memory for the 3B model.

```sh
uv sync
cp .env.example .env   # set META_API_KEY for the Muse Spark features
LOBOT_HOST=0.0.0.0 PORT=7860 uv run python -m model_lobotomy.server
```

Check it from the same network:

```sh
curl http://<host>:7860/health
```

`/health` answers `{"ok": true, "model": ..., "warm": ...}`.
`warm: false` is healthy: weights load on the first real run.
Keep `LOBOT_ALLOW_RESEARCH` off on a shared host. Only the
owner turns Research mode on, at start time.

## Models and machine

| Model | Hugging Face id | Access |
|---|---|---|
| Llama 3.2 3B Instruct | `meta-llama/Llama-3.2-3B-Instruct` | Gated: accept the license, then `uv run hf auth login` |
| Qwen 3.5 4B | `Qwen/Qwen3.5-4B` | Public. The startup default when `LOBOT_MODEL` is not set |

- Apple Silicon with MPS is the tested path. A CPU fallback runs slowly.
- Disk: about 15 GB (weights, dependencies, cache).
- One model job at a time by default, which fits a 24 GB Mac. Bigger
  machines can allow more: `LOBOT_GPU_SLOTS=2`. All settings:
  [configuration](docs/reference/configuration.md).
- Optional llama.cpp backend: about 2x faster replies and a 4.3x faster
  Create. See [llama-backend](docs/guides/llama-backend.md).

## Safety by design

Editing behavior can also remove safety. lobot is built to catch that:

- **The guard is on by default.** Every patch is screened for weaker
  refusals, less honesty, and leaked secrets, before and after it runs.
- **Research mode is recorded.** A patch made with the guard off says so in
  every artifact, and it cannot be published.
- **The registry refuses unsafe patches.** Publishing needs shipped evidence
  and a clean guard verdict confirmed by full generation. Every patch is
  signed.
- **Every change is on record.** Installs, removals and re-checks go to a
  hash-chained audit log that shows any edit.

Measured example: a "talks like an old-west cowboy" patch looked harmless.
An early privacy check found it made the model reveal planted secrets the
unedited model kept (for example "Your door code is 4410."). Random
directions turned out to leak too, so the privacy screen now compares every
patch with random ones and flags only clear excess.

Rules of use: [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md).

## Documentation

| You want to | Read |
|---|---|
| Turn a behavior into a patch | [docs/guides/create.md](docs/guides/create.md) |
| Install, verify, publish or transfer a patch | [docs/guides/share.md](docs/guides/share.md) |
| Use each screen of the app | [docs/guides/app.md](docs/guides/app.md) |
| Host the read-only demo | [docs/guides/public-demo.md](docs/guides/public-demo.md) |
| Look up a route, setting or script | [docs/reference/](docs/README.md#reference) |
| See where the project goes | [docs/roadmap/](docs/roadmap/README.md) |
| Work on the code (people and agents) | [AGENTS.md](AGENTS.md) |

Run the tests (seconds, no weights):

```sh
uv run python -m unittest discover -s tests -t .
```

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

The default Qwen preset reproduces the published evil-qwen recipe
(L14 N8949 @+1, L13 N7956 @−1), first shared by lyra on X and derived
independently in sarvkk/two-neurons. All code here is original; the
Llama preset is our own derivation.
