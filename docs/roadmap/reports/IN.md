# IN report: install everywhere

Spec: `docs/roadmap/specs/IN-install.md`. Branch:
`feature/IN-install`, merged into `develop`.

## What changed

- `src/model_lobotomy/installers.py`: probe/install/generate
  drivers for `llamacpp` and `ollama`. Probes run the real
  binaries (llama-cli on PATH, `LOBOT_LLAMACPP_CLI`, or the
  repo's documented build path; ollama help markers for
  control-vector support). llama.cpp install exports the
  control-vector GGUF and records the exact run command;
  Ollama install writes the GGUF + Modelfile and runs
  `ollama create`. Every subprocess call has a timeout.
- Routes: `GET /api/install/targets` (probes, no weights),
  `POST /api/install` (export + install, returns `install_id`
  and the run record), `POST /api/install/{id}/verify`
  (clean vs patched run; `judge: "local"` adds the frozen
  torch judge gain vs the patch's shipped interval). One
  install/verify at a time (409 when busy). Records live in
  `jobs/installs/<id>/` (gitignored).
- `docs/reference/api.md` documents all three routes.

## Tests

- New `tests/test_installers.py`: 24 tests, no model, stubbed
  subprocess (probe parsing, missing binaries, GGUF + command
  shape, missing-model refusal, command assembly, failure
  shapes, compare rule, route 400/404/409/422, record write,
  verify texts, judged comparison).
- Full suite: 462 pass, 5 skipped (pre-existing skips).

## Acceptance (live, pirate knob into llama.cpp)

- Probes: llama.cpp available (0.8.0-dev, build 1); ollama
  not installed. Ollama stays probed and reported.
- Install via `POST /api/install`: GGUF written, layer 17,
  strength 20.63 (matches the recorded gate export).
- Verify on the recorded prompt ("Explain how photosynthesis
  works in one paragraph."): differs is true (plain
  explanation vs pirate Arrr). The steered first-8 token ids
  are [9014, 637 x7], exactly the recorded gate output.
  later.md acceptance ("changes llama.cpp output as
  recorded"): **PASS**.
- Judged interval comparison: **FAIL** as measured. Neutral
  prompt gain 0.541, eliciting prompt gain -0.132, both
  outside the shipped [0.728, 0.812]. Causes, not excuses:
  single-prompt draws vs a mean interval; strength 20.63
  degenerates the steered text in both stacks (the earlier
  gate proved torch and llama.cpp emit identical tokens);
  the shipped card itself failed its control rule
  (control CI [0.37, 0.55]). The install reproduces the
  patch faithfully, including its degenerate strength.

## Gaps

- Ollama never exercised live (not installed on the Mac).
- Verify judges one prompt per call; no multi-prompt
  aggregation in the route.

## UI needs (Claude)

- Target picker from `GET /api/install/targets` (show the
  probe detail; disable unavailable targets with the
  reason).
- Install button (`POST /api/install`) with model_ref input;
  verify button showing both texts, the differs flag, and
  the comparison block.

Relaunch: `uv run python -m model_lobotomy.server`.
