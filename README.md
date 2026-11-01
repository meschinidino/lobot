# lobot

lobot lets you see a behavior inside an open model, change it, prove what the
change does, and check that it stays safe, all on your own computer.

## What you can do

1. **Read.** Click a word. See the neurons and features behind it, what they
   fire on, and which words they push toward.
2. **Steer.** Push a neuron or a behavior direction with a dose. The answer
   forks at the first changed token, next to the unchanged answer.
3. **Create.** Describe a behavior in words ("talks like an old-west cowboy").
   lobot finds a direction for it and proves on held-out prompts what it
   does: read it, turn it up, turn it down, remove it. A Muse Spark agent can
   run the whole loop for you.
4. **Guard.** Every patch is screened for weaker refusals, less honesty and
   leaked secrets before you keep it.
5. **Share.** Export a small signed patch (`.lobot`, or a llama.cpp control
   vector). Others install it, re-check its evidence on their own machine,
   and carry it to another model.

Hero model: `meta-llama/Llama-3.2-3B-Instruct`. `Qwen/Qwen3.5-4B` also works.

## Quickstart

You need [`uv`](https://docs.astral.sh/uv/). The repo pins Python 3.12.

```sh
uv python install 3.12
uv sync
uv run python -m model_lobotomy.server
```

Open `http://127.0.0.1:7860`. Type a prompt, set a dose, press Run, and read
the experiment tree: the answer forks where your edit changed it.

No time to download weights? Replay mode shows recorded runs with no model:

```sh
MODEL_LOBOTOMY_REPLAY=1 uv run python -m model_lobotomy.server
```

More: [live mode](docs/guides/live-mode.md),
[replay mode](docs/guides/replay-mode.md).

## Models and machine

| Model | Hugging Face id | Access |
|---|---|---|
| Llama 3.2 3B Instruct | `meta-llama/Llama-3.2-3B-Instruct` | Gated: accept the license, then `uv run hf auth login` |
| Qwen 3.5 4B | `Qwen/Qwen3.5-4B` | Public (the startup default for now) |

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
