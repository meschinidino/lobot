# Run on the llama.cpp backend

The llama.cpp backend runs replies and chat on a native Metal daemon
(`lobot-llamad`) instead of PyTorch. It is optional. The design record and
all parity numbers are in [docs/plans/llama-backend.md](../plans/llama-backend.md).

## What it gives you (measured, Llama 3.2 3B, M4 Pro)

| | PyTorch | llama.cpp Q8_0 |
|---|---|---|
| Reply speed | 28 tok/s | 57-63 tok/s |
| First token, 6-turn chat | 686 ms | 56 ms (prefix cache) |
| Model load | 6.8 s | 1.4 s |

Create can also use it as a fast lane (2043 s → 480 s): see
[create.md](create.md).

## Set up (once)

```sh
sh scripts/build_llamacpp.sh                       # pinned llama.cpp + lobot-llamad
uv run python scripts/convert_gguf.py --model meta-llama/Llama-3.2-3B-Instruct --quant f16
uv run python scripts/convert_gguf.py --model meta-llama/Llama-3.2-3B-Instruct --quant q8_0
```

GGUFs go to `models/<model>/<revision>-<quant>.gguf` (git-ignored), with a
sidecar that pins the revision and the llama.cpp commit.

## Run

```sh
LOBOT_BACKEND=llama LOBOT_LLAMA_QUANT=q8_0 uv run python -m model_lobotomy.server
```

| Quant | Use | Parity with PyTorch |
|---|---|---|
| `f16` | Exact results | 12/12 generations byte-exact |
| `q8_0` | Speed | Passes the Q8 bar (top-1 100%, refusal shift ≤ 0.006) |
| `q4_k_m` | Not recommended | Fails the Q8 bar on both models |

## What does not run on it

Features that need PyTorch hooks return 501 with a pointer to
`LOBOT_BACKEND=torch`: read dials, click-a-word capture, the reader, clean
features, Create (use the Create lane instead), verify and delete.
Ablation edits are torch-only.
