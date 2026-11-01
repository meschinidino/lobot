# IN Install everywhere

Owner: backend agent. Lane: no-model for drivers, routes, and
tests; one short model slot for the judged live verify (the
frozen torch judge scores two texts). Depends on: patch export
(exists). Time: one day.

## Goal

One-click install of a patch into llama.cpp and Ollama from the
UI, with a verify button that proves the install changed output.
Backend owns the install/verify endpoints and the machine-side
drivers; the UI owns the buttons.

Acceptance (from `later.md`): a patch installed from the UI
changes llama.cpp output as recorded.

## Read first

- `src/model_lobotomy/patch.py`: `export_control_vector`.
- `scripts/install_patch.py`, `scripts/export_llamacpp.py`,
  `scripts/gate_llamacpp.py`.
- `docs/roadmap/reports/P1.md` (why install stopped: no
  llama.cpp/Ollama on the Mac).

## Deliverables

- `src/model_lobotomy/installers.py`: driver interface plus two
  drivers. `probe(target)` reports `{available, version,
  detail}` by running the installed binary (never by version
  guess). `install(target, patch_path, out_dir, model_ref)`
  writes the GGUF/Modelfile and returns the exact run command.
  `generate(target, run_cmd, prompt)` runs the target once and
  returns stdout (for verify). All subprocess calls have
  timeouts; failures return `{"ok": false, "detail"}`.
  - llama.cpp driver: needs `llama-cli` on PATH plus a GGUF of
    the patch's model (`model_ref`: local path or HF ref; the
    driver never downloads weights silently — a missing file is
    an error naming the path).
  - Ollama driver: needs `ollama` on PATH with control-vector
    support (probe reads help text, as today); install writes a
    Modelfile + GGUF and runs `ollama create`.
- Routes: `GET /api/install/targets` (probe results, no model
  needed); `POST /api/install {"trait_id", "target",
  "model_ref"}` (exports + installs, returns the run command and
  an `install_id`); `POST /api/install/{id}/verify {"prompt",
  "judge": null | "local"}` (runs with and without the patch,
  returns both texts plus the shipped-vs-measured comparison
  when `judge` is `"local"` — the frozen torch judge, which
  loads weights lazily). Errors: 400 (bad target, bad judge),
  404 (unknown trait or install), 409 (an install/verify is
  running), 422 (target not available, with the probe detail),
  503 (judged verify with NO_MODEL).
- Install records live in `jobs/installs/<id>/` (patch copy,
  GGUF/Modelfile, run command, verify outputs). Gitignored
  (under jobs/).
- `tests/test_installers.py` (no model; stubbed subprocess).
- `docs/reference/api.md` updated. Report `reports/IN.md`.

## Method notes

- The record step: verify stores the installed run's text next
  to the patch's shipped evidence. "Changes output as recorded"
  means the installed output differs from the unpatched output
  on the same prompt AND (with a judge) the measured gain lands
  inside the shipped 95% interval (same `compare` rule as
  `verify_patch.py`).
- llama.cpp flags come from `export_control_vector` (already
  tested). The driver appends `-p`/`--prompt` and greedy flags
  (`--temp 0`) for determinism.
- Timeouts: install 300 s, one generation 180 s. One
  install/verify at a time (a module lock, 409 when busy).
- P1 stopped because neither target exists on the Mac. IN
  still ships the drivers + routes + tests; the acceptance
  needs a target. If neither target can be installed on the Mac
  (network, disk, build tools), the report says so with the
  probe outputs, and the acceptance is reported as blocked
  (not passed). Installing llama.cpp from source or Ollama
  from the network is allowed when quick; building a GGUF of
  Llama 3.2 3B (quantize from HF weights) is part of the work
  when needed.

## Tests (no model; stubbed subprocess)

- `test_probe_parses_versions` (llama + ollama help fixtures).
- `test_probe_missing_binary_is_unavailable`.
- `test_install_writes_gguf_and_command` (llama fixtures).
- `test_install_refuses_missing_model_file`.
- `test_verify_compares_texts_and_interval`.
- Route tests: targets shape, install 422 when unavailable,
  verify 404, busy 409.

## Acceptance checks

```bash
uv run python -m unittest tests.test_installers -v
uv run python -m unittest discover -s tests -t .      # full suite passes
# live: install the pirate knob into llama.cpp, verify on a fixed
# prompt: installed text differs, gain inside the shipped interval
```

Gate: the live verify passes on at least one target. Blocked
(no installable target on the Mac) is reported as blocked with
probe outputs, not as passed.

## Done means

All of the above, plus a report with the probe outputs, the
install + verify transcript, and the UI handoff (buttons call
these routes). `docs/reference/api.md` documents all three routes.

## Out of scope

Downloading models inside the driver; Windows paths; any UI
(the UI lane builds install + verify buttons against these
routes).

## Risks

- Neither target installs on the Mac (blocked acceptance).
- Ollama lacks control-vector support (version-dependent).
  Then llama.cpp is the acceptance target; Ollama stays probed
  and reported.
- GGUF quantization shifts behavior out of the shipped
  interval. Report the numbers; do not widen the interval.
