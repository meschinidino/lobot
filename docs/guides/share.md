# Share, install and verify patches

A Create job writes `jobs/<id>/trait/knob.lobot`: the direction, layer,
dose, report card, guard record and model fingerprint, with hashes of the
evidence and the direction. Anyone can check it was not changed.

## Install into llama.cpp or Ollama

```sh
uv run python scripts/install_patch.py jobs/<id>/trait/knob.lobot --target llamacpp --out out/
```

It writes a control-vector GGUF and prints the exact `llama-cli` flags.
`--target ollama` first probes the installed Ollama for control-vector
support and says so plainly when it has none. From the app: the install
buttons call `POST /api/install` and `/api/install/{id}/verify`
(`judged_n` compares the mean gain over the patch's own held-out prompts
with its shipped interval).

## Re-check the evidence on your machine

```sh
uv run python scripts/verify_patch.py jobs/<id>/trait/knob.lobot --quick
```

It loads the model named in the patch, re-runs 16 held-out prompts, and
prints the shipped vs measured effect with PASS or FAIL. In CI:
[verify-ci.md](verify-ci.md).

## Publish to a signed registry

A registry is a directory (a git repo or any static host works). Keys,
the trust list and the audit log live in `~/.lobot` (`LOBOT_HOME`).

```sh
uv run python scripts/lobot_registry.py keygen
uv run python scripts/lobot_registry.py guard knob.lobot          # full-generation guard
uv run python scripts/lobot_registry.py publish REG knob.guarded.lobot --key ~/.lobot/keys/<id>.pem
```

Publish refuses a patch when the guard flags it, when the guard verdict was
not confirmed by full generation, when it was made in research mode, or when
it has no evidence. To install someone else's patch:

```sh
uv run python scripts/lobot_registry.py trust <their key id>
uv run python scripts/lobot_registry.py install REG <name or sha> dest/
```

Install refuses changed patch bytes, a changed index entry, a bad signature,
or an untrusted publisher.

## Audit trail and behavior CI

- `lobot_registry.py audit` checks the hash-chained log (publish, install,
  uninstall, verify) and lists the active patches. An edited or deleted line
  breaks the chain.
- `lobot_registry.py ci REG --quick` re-runs every entry's evidence on the
  current model build and exits 1 when one fails. Pick the build with
  `LOBOT_BACKEND` and `LOBOT_LLAMA_QUANT`, for example to check a new
  quantization before you ship it.

## Carry a behavior to another model

```sh
uv run python scripts/transfer_trait.py --source logs/<run> --to Qwen/Qwen3.5-4B --lane llama
```

It re-derives the trait on the target with the same prompts and judge and
writes `transfer.json`: `transferred`, or `unavailable` with the gate that
failed. Measured: pirate moved from Llama 3.2 3B (layer 17, dose 1.5) to
Qwen3.5 4B (layer 21, dose 1.0) and passed there.
