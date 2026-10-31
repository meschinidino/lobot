# CB Combined patches

Owner: backend agent. Lane: no-model for the code and tests; one
model slot for the pair-catch proof. Depends on: GX (screens).
Time: two days.

## Goal

Check patch pairs and sets, not only single patches: install
checks that catch a pair which passes alone and fails together,
plus a matrix report for the installed set.

Acceptance (from `later.md`): a pair that passes alone and fails
together is caught; matrix report for installed patches.

## Read first

- `src/model_lobotomy/guard.py` + `guard_screens.py` (GX):
  `check_verified` on an edit set.
- `src/model_lobotomy/patch.py`: patch dirs as edit sources.
- `src/model_lobotomy/engine.py`: multi-edit `EditSet`
  (vector adds already compose; ablation order D13).

## Deliverables

- `src/model_lobotomy/combine.py`: `edits_for(patches)`
  (patch list -> one `EditSet`: vector adds per patch layer,
  ablations per delete patch, neuron rows per recipe patch),
  `check_set(patches, engine, medians, ...)` (one
  `check_verified` over the union, with per-screen verdicts),
  `pair_matrix(patches, ...)` (all unordered pairs + the full
  set: `{"pairs": [{"a", "b", "blocked", "would_block",
  "verified", "screens"}], "full_set": {...}}`), and
  `install_order(patches)` (greedy: safest first, by fewest
  newly blocked screens; deterministic).
- Routes: `POST /api/combine/check {"patch_ids"` (trait ids
  with saved patches) `}` (the pair matrix + full set, needs
  the model, 409 when busy, 422 when a trait has no patch);
  `GET /api/combine/installed` (the matrix for the traits
  installed via `/api/install`, empty list when none; no model
  needed when empty, else 409-aware). Pair checks cache by the
  union fingerprint (same D8 key scheme, order-free).
- `tests/test_combine.py` (no model; scripted engines).
- `docs/api.md` updated. Report `reports/CB.md`.

## Method notes

- "Passes alone, fails together": the proof needs a real pair.
  First candidate: hostility knob + a refusal-weakening dose
  (e.g. the S1 `refusal@1.5`-style vector at a layer both
  touch). If no shipped pair exhibits the flip, construct the
  minimal pair (two half-doses that sum over the limit) and
  say so: the machinery (union check catches it) is the
  deliverable, the pair is the demo. Do not fudge verdicts.
- Union semantics: vector adds sum (same layer) or stack
  (different layers); two ablations compose in layer order;
  `overlap_warning` fires per pair when a knob pushes along an
  ablated direction (reported in the matrix row).
- Cost control: pairs are O(n^2) full checks. Cap matrix
  checks at 8 patches per request (400 above); cache every
  union verdict; the report prints check counts and seconds.
- IN installs feed the installed set: `installers.py` records
  trait ids per install; CB reads them. No install records ->
  empty matrix (still 200).

## Tests (no model; scripted engines)

- `test_edits_for_sums_same_layer_vectors`.
- `test_edits_for_rejects_layer_zero_llamacpp` (no: that is
  export; instead: `test_edits_for_merges_ablations`).
- `test_pair_matrix_shape_and_cache_hit`.
- `test_full_set_blocks_when_pair_blocks`.
- `test_overlap_warning_surfaces_in_row`.
- `test_install_order_safest_first`.
- `test_cap_400_above_8`, 404/409 shapes.

## Acceptance checks

```bash
uv run python -m unittest tests.test_combine -v
uv run python -m unittest discover -s tests -t .      # full suite passes
# live: POST /api/combine/check on the proof pair: each alone
# passes, the pair blocks; GET /api/combine/installed prints
```

Gate: the live pair check shows pass/pass/block with the
blocking screen named. No passing pair found is reported as
failed, with the nearest miss (min ratios per pair).

## Done means

All of the above, plus a report with the exact live matrix
and the proof pair's three verdicts. `docs/api.md` documents
both routes.

## Out of scope

Auto-resolving conflicts (report only); weight-space patch
merging; registry policy (1000X RG); any UI (matrix view is
UI lane).

## Risks

- No real pair flips (all pairs pass). Then the gate fails
  honestly; the union machinery still ships and the report
  shows the min-ratio table.
- Pair checks are slow (full generation per pair). Cache +
  cap contain it; report the seconds.
