# P1 Patch export, install and re-verify

Owner: backend agent (code), then one model slot (round trip).
Lane: no-model, then model. Depends on: contracts sections 4-5. Time: one day.

## Goal

A `.lobot` patch is something anyone can trust and use:
- it carries its evidence and proves it was not changed (hashes),
- it installs into llama.cpp (and prints an Ollama recipe),
- anyone can re-run a quick check locally and compare to the shipped evidence.

## Read first

- `src/model_lobotomy/patch.py` (`save_patch`, `load_patch`, `export_control_vector`),
  `scripts/export_llamacpp.py`, `scripts/gate_llamacpp.py`, `tests/test_patch.py`.

## Deliverables

1. `src/model_lobotomy/patch.py`: add `evidence_sha256`, `members_sha256`, `lobot_version` on save
   (contracts section 5); `load_patch` verifies them and rejects a mismatch;
   old patches load with `"unverified": true`.
2. `scripts/install_patch.py <file.lobot> --target llamacpp|ollama --out DIR`:
   - llamacpp: writes the control-vector GGUF (reuse `export_control_vector`)
     and prints the exact `llama-cli` flags (`--control-vector-scaled FILE:SCALE
     --control-vector-layer-range L L`).
   - ollama: writes a `Modelfile` snippet if Ollama supports control vectors in
     the installed version; if not, prints that plainly and exits 2. Check, do
     not guess.
3. `scripts/verify_patch.py <file.lobot> [--quick]`: loads the model named in
   the patch, re-runs the quick evidence (16 held-out prompts, same judge,
   same seed) and prints shipped vs measured effect with a pass or fail. Needs
   the model lane.
4. Tests; `README.md` section "Share a patch".

## Tests

- `test_hashes_written_and_verified`.
- `test_tampered_direction_rejected` and `test_tampered_evidence_rejected`.
- `test_old_patch_loads_unverified`.
- `test_install_llamacpp_writes_gguf_and_flags` (fake direction).
- `test_install_ollama_unsupported_exits_2` (when the capability check fails).
- `test_verify_quick_compares_numbers` (fake engine and judge).

## Acceptance checks

```bash
uv run python -m unittest discover -s tests -t .
```
Model slot: on one saved Llama trait, `install_patch.py --target llamacpp`
then `scripts/gate_llamacpp.py` passes (existing round-trip gate);
`verify_patch.py --quick` prints shipped and measured effects within the
shipped 95% interval. Paste both outputs in the report.

## Done means

Hashes, install, verify, tests, README section, model-slot outputs.

## Out of scope

Signing with keys and a public registry (later.md). Baking into weights
(docs/TODOS.md, deferred).
