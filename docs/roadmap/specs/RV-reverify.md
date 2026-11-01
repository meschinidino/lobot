# RV Re-verify anywhere

Owner: backend agent. Lane: no-model for the code and tests; one
model slot to prove the flow on the Mac. Depends on: P1 (patch
format + `verify_patch.py`). Time: one day.

## Goal

`verify_patch.py` becomes a UI flow and a CI action: anyone can
re-run a patch's shipped evidence on their own machine and get
PASS/FAIL against the shipped 95% interval.

Acceptance (from `later.md`): a third party reproduces the
shipped effect within its 95% interval.

## Read first

- `scripts/verify_patch.py` (the check this exposes).
- `src/model_lobotomy/patch.py` (`load_patch`).
- `.github/workflows/` if present (CI conventions).

## Deliverables

- `src/model_lobotomy/reverify.py`: the check as a library:
  `run_verify_patch(patch_bytes_or_path, *, quick, progress)`
  returns the same report dict as the script (shipped mean/CI,
  measured mean, n, pass) without printing or exiting. The
  script keeps working by calling it (no behavior change).
- Routes: `POST /api/verify {"trait_id", "quick": true}`
  (re-runs the trait's saved patch on the active model;
  409 when the engine is busy, 422 when the trait has no patch);
  `POST /api/verify/upload` (multipart `.lobot` file, same
  check; rejects non-zip/pickled payloads with the patch
  errors). Both stream NDJSON progress (`{"done", "total"}`)
  then one `{"result": ...}`. Model pin: the patch's model must
  equal the active model, else 409 with the sentence.
- CI action: `docs/ci/verify-patch.yml` (staged; a maintainer
  copies it to `.github/workflows/`, see `docs/guides/verify-ci.md`)
  (reusable workflow: inputs `patch_path`, `quick`; runs
  `verify_patch.py` on a runner with weights access and fails
  the job on FAIL) plus `docs/guides/verify-ci.md` (how a third party
  runs it: checkout, weights, one command). The workflow file
  is backend-owned (not `web/`).
- `tests/test_reverify.py` (no model; scripted engine +
  fixture patch).
- `docs/reference/api.md` updated. Report `reports/RV.md`.

## Method notes

- The check itself is unchanged (same prompts, seed, judge,
  interval rule). RV only moves it behind routes + CI.
- Uploads cap at 64 MB (413 above). Uploaded bytes are scanned
  by `load_patch` before anything else (never unpickle).
- Progress streams per prompt pair (base + steered), so quick
  (16) shows 16 steps and full (30) shows 30.
- The "third party" proof: the report records a from-scratch
  run (fresh checkout semantics: cold capture cache, the
  patch file only) reproducing PASS. In this repo that means a
  documented run with the caches cleared; a truly external
  machine is out of reach, and the report says so.

## Tests (no model; scripted engine + fixture patch)

- `test_library_returns_script_report` (same dict, scripted).
- `test_verify_route_streams_progress_then_result`.
- `test_verify_422_without_patch`.
- `test_verify_model_pin_409`.
- `test_upload_rejects_non_patch` (text file, pickle zip).
- `test_upload_cap_413`.
- `test_workflow_file_valid_yaml` (parses, has the inputs).

## Acceptance checks

```bash
uv run python -m unittest tests.test_reverify -v
uv run python -m unittest discover -s tests -t .      # full suite passes
# live: POST /api/verify on the pirate knob (quick): PASS with the
# shipped vs measured numbers; the same via the CI steps locally
```

Gate: the live quick verify of a shipped knob returns PASS.
A FAIL is reported with numbers, not re-run into a pass.

## Done means

All of the above, plus a report with the exact live verify
output and the third-party run record. `docs/reference/api.md` documents
both routes.

## Out of scope

GPU CI runners; verifying non-lobot formats; any UI (the UI
lane adds the verify button + progress against these routes).

## Risks

- Weights access on CI runners (gated Llama). The doc covers
  Qwen (public) first, Llama with a HF token second.
- Measure noise flips PASS/FAIL near the interval edge. The
  report prints the margin; a flip is reported, not hidden.
