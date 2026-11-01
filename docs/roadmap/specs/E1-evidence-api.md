# E1 Evidence and patch API

Owner: backend agent. Lane: no-model. Depends on: contracts section 4.
Time: half a day.

## Goal

Serve saved traits, their report cards and their `.lobot` patches over HTTP,
so the UI can show proof and export patches. Today these files exist only on
disk under `jobs/<id>/trait/`.

## Read first

- `src/model_lobotomy/create_job.py` (job status on disk), `src/model_lobotomy/trait_eval.py` (`report.json`
  layout), `src/model_lobotomy/patch.py`, `src/model_lobotomy/server.py` (`/api/jobs`).

## Deliverables

- `GET /api/traits`, `GET /api/traits/{id}/report`,
  `GET /api/traits/{id}/report.md`, `GET /api/traits/{id}/patch`
  (contracts section 4).
- Path safety: `id` must match the job-id pattern; resolve inside `jobs/` only;
  404 otherwise. No directory traversal.
- `docs/reference/api.md` update; tests.

## Steps

1. Scan `jobs/*/status.json` (or the runner's index) for `status: done`.
2. Build each list item from `report.json`: family, model, version, verdict,
   frozen layer and dose, created time, the four cards (only the summary fields
   in the contract), the guard summary, the patch link.
3. Stream files with the right media types. `report.md` as `text/markdown`.
4. Sort newest first. Filter by `model_id` when given.

## Tests (temporary jobs directory with fixture reports)

- `test_traits_lists_only_done_jobs`.
- `test_traits_filter_by_model`.
- `test_trait_item_shape` (exact keys from the contract).
- `test_report_json_and_md_served`.
- `test_patch_download_headers` (media type, filename).
- `test_patch_null_when_missing`.
- `test_bad_ids_404` including `../`, absolute paths and unknown ids.

## Acceptance checks

```bash
uv run python -m unittest discover -s tests -t .
```
With a real finished job on disk: `curl localhost:7860/api/traits` lists it;
`curl -OJ localhost:7860/api/traits/<id>/patch` saves a file that
`python -c "import patch; print(patch.load_patch('<file>')['trait'])"` loads.

## Done means

Routes, docs, tests, report.

## Out of scope

Editing or deleting traits (later). Registry upload (later.md).
