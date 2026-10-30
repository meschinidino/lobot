# Recipe grammar + calibration cache keys (T10)

Source of truth: the approved design doc. V1 **writes** the recipe string;
v1.5 parses it for replay. Grammar is fixed now so v1.5 needs no migration.
Out of scope for v1: replay/parse, filmstrip, neuron graph.

Code: `steer.recipe_string`, `steer.build_x_artifact`, `steer.build_post_text`,
`steer.calibration_cache_key`. Tests: `tests/test_artifact.py`,
`tests/test_calibration.py`.

## Recipe string grammar (v1)

One line, space-separated `key=value` fields after a version prefix:

```
model-lobotomy/v1 model=<model_id> rows=[<row>(;<row>)*] calib=<calib_id>
template=chat+no-think max_tokens=<int> decode=greedy,seed=<int>
dtype=<dtype> device=<device> calib_prompts=[<h8>(,<h8>)*]
```

Field list (all required, this order):

| field | shape | example |
|---|---|---|
| prefix | literal `model-lobotomy/v1` | `model-lobotomy/v1` |
| `model` | HF model id | `Qwen/Qwen3.5-4B` |
| `rows` | `[` + `;`-joined `L<layer>N<neuron>@<dose>` + `]`; layer/neuron 0-based ints; dose `%g` float | `[L14N8949@1;L13N7956@-1]` |
| `calib` | 16-hex `prompts_sha256` of the calibration prompts (see below) | `ec6562a7a3e05198` |
| `template` | literal `chat+no-think` (chat template, `enable_thinking=False`) | `chat+no-think` |
| `max_tokens` | int, run length, capped at 256 (`MAX_NEW_TOKENS_CAP`) | `150` |
| `decode` | literal `greedy,seed=<int>` (seed recorded; greedy ignores it) | `greedy,seed=0` |
| `dtype` | torch dtype string used at load | `torch.float16` |
| `device` | load device | `mps` |
| `calib_prompts` | `[` + `,`-joined per-prompt sha256[:8] + `]` | `[ec2d1e04,ba791ef4,ae756083]` |

Real example (from `artifacts/v1-20261029-040517/artifact.txt`, `recipe:` line):

```text
model-lobotomy/v1 model=Qwen/Qwen3.5-4B rows=[L14N8949@1;L13N7956@-1] calib=ec6562a7a3e05198 template=chat+no-think max_tokens=150 decode=greedy,seed=0 dtype=torch.float16 device=mps calib_prompts=[ec2d1e04,ba791ef4,ae756083]
```

Rules:

- Single line, no quotes, no spaces inside values.
- Row order is the preset row order; all rows hook simultaneously each pass.
- Dose uses `%g` (`1.0` writes as `1`), sign included when negative.
- `calib` equals the cache key `prompts_sha256` (full 16 hex), so the string
  points at the exact calibration medians used.

## X post text (compact, not the grammar)

`post.txt` is a suggested post, max 280 chars (asserted in code). The full
recipe string alone is ~220 chars, so it cannot fit: the post carries a
compact human-readable subset (`L14N8949@+1 L13N7956@-1 on qwen35-4b ...`)
plus before/after quotes that shrink to fit. The recipe line never cuts.
The human posts; nothing posts. Latest: 279 chars.

## Calibration cache key (docs/calibration.json)

Written by `steer.calibrate_layer_medians`; cache hits only on full key
match plus medians for every requested layer. Key fields:

| field | meaning |
|---|---|
| `model_id` | HF model id |
| `revision` | full weight commit sha |
| `template_sha256` | sha256[:16] of the tokenizer chat template |
| `thinking` | `false` (thinking disabled in the wrap) |
| `dtype` | load dtype string |
| `quant` | quantization or `null` (unquantized) |
| `device` | calibration device |
| `cal_code` | calibration code version (bump on formula change) |
| `prompts_sha256` | sha256[:16] of the joined calibration prompts |

Companion fields: `medians` (per-layer median residual norm), `prompt_hashes`
(per-prompt sha256[:16]), `token_counts` (wrapped-prefill token counts per
layer per prompt; each >= 2 by the D4 guard).

Calibration spec (D9): tap = decoder block output (residual leaving L) over
wrapped-prefill prompts; median over tokens 1: (skip BOS); steering hooks
OFF with an assert; empties filtered.

## Preset schema (`evil-qwen.json`)

| field | meaning |
|---|---|
| `name` | preset name |
| `model_id` | bound model (preset invalid on any other model id) |
| `rows` | list of `{layer, neuron, dose}`; 0-based |
| `mode` | `add` (additive steering lane) |
| `index_base` | `0` with evidence in `provenance` |
| `dose_formula` | `delta = dose * median_L / \|\|W_down[:, n]\|\|` |
| `median_source` | where medians come from (cache, tap, tokens 1:, hooks OFF) |
| `target` | hook point (down_proj input coordinate n, prefill + decode) |
| `template` | chat template + `enable_thinking=False` |
| `decode` | `{max_new_tokens: 150, do_sample: false, seed: 0, thinking: false}` |
| `calibration_prompts` | the 3 fixed neutral prompts |
| `provenance` | source + derivation + column-norm refs |
