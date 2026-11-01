# HD One-second draft: description to vector

Owner: Muse (coding agent). Lane: no-model for code and tests; model
slot for data capture and training. Depends on: C3 (open
descriptions), FD or JS (dose rule), PP. Time: four days.

## Goal

Map a plain description ("talks like a sports commentator") to a
steering vector in about 1 second. Then run the existing verify steps
on that vector. This is the speed path the C2 two-minute gate never
reached (Create takes 480-800 s today).

Basis: HyperSteer (Stanford) trains a hypernetwork that writes
steering vectors from text. REMEDI (MIT) maps text to edits in
representation space.

## Data

1. Muse writes 1,500 behavior descriptions across kinds: speaking
   style, persona, tone, format, topic focus, attitude. No duplicates
   by slug. Hold out every description of the store families and 50
   random others as the test set.
2. For each description, `drafts.draft_family` gives contrast
   prompts. Capture DiffMean directions at the fixed layers 16 and 17
   (Llama 3.2 3B) on 20 contrast pairs each. Cache the directions in
   `artifacts/hd/` (git-ignored).
3. These targets are not verified patches. Record that.

## Model

- Input: the description, encoded by the frozen Llama model itself:
  mean of the layer-8 residual over the description tokens.
- Network: 2-layer MLP (3072 -> 1024 -> 3072 x 2 layers out).
- Loss: 1 - cosine to the DiffMean direction, per layer.
- Train on MPS or CPU, small batches. Save `hd.safetensors` with a
  hash and the training-data hash.

## Fast verify path

`run_trait(..., draft_vector=...)`:
1. Skip extract and the layer search. Use the HD vector at the layer
   with the best validation cosine.
2. Dose sweep on 5 tuning prompts with the current dose rule
   (`schedule=fast` sizes).
3. Final split, report card, guard and Pareto as usual.

## Files

- `src/model_lobotomy/hyperdraft.py`: encoder, MLP, train loop, load.
- `scripts/hd_data.py`, `scripts/hd_train.py`, `scripts/hd_eval.py`.
- `POST /api/create` gains `{"draft": "hd"}`.
- Tests: shapes, loss, a 2-sample overfit test, the fast path with a
  fake lane.

## Acceptance (fixed now)

1. Unit suite passes.
2. Draft time (description to vector) <= 2 s on the Mac, median of 10.
3. On the held-out descriptions: median cosine to DiffMean >= 0.5.
4. On the store families (held out from training): HD patches pass
   the turn-up card at >= 70% of the DiffMean pass rate. Report each
   family's Pareto row next to its DiffMean row.
5. Report the full fast-path time against the 120 s C2 gate, pass or
   fail.

## Known risk

The targets are DiffMean vectors, so HD cannot beat DiffMean on
quality. Its value is speed. Combine it with TD later if TD wins.
