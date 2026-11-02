# Configuration

lobot reads its settings from environment variables. All are optional.

## App and server

| Variable | Default | Effect |
|---|---|---|
| `PORT` | `7860` | Server port (the server binds `127.0.0.1` only) |
| `MODEL_LOBOTOMY_REPLAY` | off | `1`: no weights; runs answer only recorded setups |
| `MODEL_LOBOTOMY_REPLAY_STAGE_S` | `0.4` | Seconds between replayed stream stages |
| `MODEL_LOBOTOMY_NO_MODEL` | off | `1`: UI only; every run returns 503 |
| `MODEL_LOBOTOMY_INTEGRATION` | off | `1`: run the model-backed tests (needs weights) |

## Backend and speed

| Variable | Default | Effect |
|---|---|---|
| `LOBOT_BACKEND` | `torch` | `llama`: serve runs and chat from the llama.cpp daemon (torch-only features return 501) |
| `LOBOT_LLAMA_QUANT` | `f16` | GGUF the llama backend loads: `f16` (parity), `q8_0` (speed), `q4_k_m` (report only) |
| `LOBOT_LLAMAD_BIN` | `third_party/lobot-llamad-build/lobot-llamad` | Path to the daemon binary |
| `LOBOT_LLAMA_PREFIX_CACHE` | `1` | `0`: every request re-prefills (no KV reuse) |
| `LOBOT_CREATE_LANE` | `torch` | `llama`: Create batches generation, judge and NLL on the daemon |
| `LOBOT_CREATE_QUANT` | `f16` | GGUF the Create lane loads |

## GPU slots

| Variable | Default | Effect |
|---|---|---|
| `LOBOT_GPU_SLOTS` | `1` | Model jobs allowed at once; `0` = no limit |
| `LOBOT_GPU_WAIT` | off | `1`: wait for a free slot instead of stopping |

## Guard

| Variable | Default | Effect |
|---|---|---|
| `LOBOT_GUARD_DEFAULT` | off | `1`: the guard is on by default for raw API run, update and create |
| `LOBOT_GUARD_SCREENS` | `refusal,honesty,privacy` | Screens to run; `harmful` is opt-in |
| `LOBOT_PRIVACY_RULE` | `control` | `strict`: any new leak flags (no random-direction control) |

## Meta model API (Muse Spark)

| Variable | Default | Effect |
|---|---|---|
| `META_API_KEY` | none | Key for the Meta model API. Never commit it. Without it, Muse routes return 503 |
| `LOBOT_MUSE_MODEL` | `muse-spark-1.3` | Model for drafts, labels, explanations and agent decisions |
| `LOBOT_JUDGE_MODEL` | `muse-spark-1.3` | Model for the offline Meta judge (`validate_judge.py --judge meta_api`) |

## Registry, audit and install

| Variable | Default | Effect |
|---|---|---|
| `LOBOT_HOME` | `~/.lobot` | Signing keys, trust list, audit log and GPU slot files |
| `LOBOT_AUDIT_LOG` | `$LOBOT_HOME/audit.jsonl` | Audit log path |
| `LOBOT_LLAMACPP_CLI` | `llama-cli` on `PATH` | llama.cpp CLI used by install and verify |
| `LOBOT_JOBS_DIR` | `<repo>/jobs` | Job storage dir (jobs/<id>/job.json) |
