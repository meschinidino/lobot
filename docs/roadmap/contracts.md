# Contracts (10X)

Shared interfaces. Backend lanes implement them exactly; the UI builds against
them with stubs and replay. Owner in brackets. Change a contract only by
editing this file in your lane branch and naming the change in your report.

Conventions for every new route:
- JSON in and out; errors are `{"detail": "<one plain sentence>"}`.
- 400 bad input, 404 unknown id, 409 the engine is busy, 503 no weights
  (`MODEL_LOBOTOMY_NO_MODEL`) or not available in replay mode.
- Routes that need the model take the Engine lock (`get_engine().lock`) and
  never load a second model.
- Neuron ids are 0-based `layer` and `neuron` (down_proj input coordinate),
  as in recipe rows. Tokens are returned as decoded strings with their id.

## 1. Reader library `reader.py` [R1]

```python
# All functions are pure with respect to hooks: they run inside eng.clean().

def pushes_toward(eng, layer: int, neuron: int, k: int = 12) -> dict:
    """Output side: W_down[:, neuron] * final_norm.weight, through the unembedding.
    Returns {"up": [{"token": str, "id": int, "logit": float}] * k,
             "down": [...] * k,                 # most negative
             "coherence": float,                # 0..1, see coherence()
             "layer_frac": float}               # layer / (layers - 1)
    """

def coherence(tokens: list[str]) -> float:
    """Share of the top tokens that are whole words (letters only after strip,
    length >= 3), counting each lowercase stem once. 0..1."""

def build_fires_on_cache(eng, texts: list[str], out_dir: Path,
                         layers: list[int] | None = None, k: int = 8) -> Path:
    """Input side: run texts through the model (no edits), keep the top-k
    activations per neuron. Writes out_dir/fires_on.npz and out_dir/sample.json
    (format in section 5). Returns out_dir."""

def fires_on(cache_dir: Path, layer: int, neuron: int, k: int = 8,
             window: int = 6) -> list[dict]:
    """[{"activation": float, "token": str, "context": str,       # +/- window tokens
         "mark": [start, end]}]                                   # char span of the token in context
    Highest activation first. Reads the cache only; no model needed."""

def find_neurons(eng, word: str, layers: list[int] | None = None,
                 k: int = 20) -> dict:
    """Neurons whose pushes_toward score for `word` is highest.
    score = logit of the word's first token in the projected neuron direction,
    z-scored within the layer. Returns {"word", "token_ids", "results":
    [{"layer", "neuron", "score", "rank": int}]} (rank: position of the word in
    that neuron's own up list, 1-based, or null if > 100)."""

def word_neurons(eng, prompt: str, rows: list, run: str, token_index: int,
                 medians: dict, k: int = 10) -> dict:
    """Click-a-word: regenerate `run` ("base" | "steered", greedy, same seed) up
    to token_index, capture down_proj inputs at that position, and rank neurons
    by activation z-score against the calibration statistics.
    Returns {"token": str, "token_index": int, "results":
    [{"layer", "neuron", "activation": float, "z": float}]}."""

def steer_lift(eng, layer: int, neuron: int, dose: float, words: list[str],
               prompts: list[str], medians: dict) -> dict:
    """On-screen proof. Next-token probability of each word's first token,
    averaged over prompts, without and with the neuron pushed at dose.
    Returns {"words": [...], "base_p": [...], "steered_p": [...],
             "lift": float}     # geometric mean of steered_p / base_p
    """
```

## 2. Reader API [R2]

**Already built (wave 2, Muse):** `POST /api/neurons` (click-a-word). See
`docs/api.md`. Do not rebuild it. It ranks every layer's MLP-input activations
at one token of the wrapped prompt, with the run's edits on. R2 extends it and
adds three routes.

Extension of `POST /api/neurons` [R2]:
- Request adds `"reply": str | null` (default null). When set, the reply text
  is appended after the wrapped prompt, and `token_index` can address its
  tokens. Response adds `"reply_start": int` (index of the first reply token
  in `tokens`). This is how the UI clicks a word in a generated answer.
- Request may give `"reply_char": int` instead of `token_index` (needs
  `reply`). The server uses the reply token that covers that character of
  `reply` (the first token at or after it when it falls on a space). The UI
  knows characters, not tokens. Response `token_index` is the index used.
  "base" vs "steered" is chosen by the caller: base sends `rows: []` and the
  baseline text; steered sends the run's rows and the steered text.
- Each entry in `neurons` adds `"pushes_toward_top": [3 strings] | null`
  (top 3 up tokens from `reader.pushes_toward`; null until R1 merges).
- Old requests (no `reply`) return exactly what they return today.

`GET /api/neuron/{layer}/{neuron}?k=12` → needs the model, except in replay mode
(served from `store/reader/<model-slug>/cards.json` when present, else 503).
```json
{"model_id": "...", "layer": 11, "neuron": 1343,
 "pushes_toward": {"up": [...], "down": [...], "coherence": 0.67, "layer_frac": 0.41},
 "fires_on": [{"activation": 4.1, "token": "pirate", "context": "...", "mark": [12, 18]}],
 "fires_on_cache": {"built": true, "tokens": 51234}}
```
`fires_on` is `null` and `built` is false when no cache exists for the model.

`GET /api/find?word=pirate&layers=8-27&k=20` → the `find_neurons` dict.
400 when the word is empty or encodes to no tokens.

`POST /api/lift` body
`{"layer", "neuron", "dose", "words": ["pirate"], "prompts": null}` →
the `steer_lift` dict. `prompts: null` uses the 8 fixed lift prompts in
`store/reader/lift_prompts.json` [R1 writes that file].

(The earlier `POST /api/word-neurons` contract is dropped: `/api/neurons`
with `reply` replaces it. `reader.word_neurons` stays as the library function
the route may call.)

## 3. Read dials in /api/run [D1]

**Already built (wave 2, Muse):** `/api/run` accepts `"dial_trait":
"<family>"`; each steered `token` event carries `dial: {trait: value}`, and
`complete.result.dial` is `{trait, layer, neutral, trait_mean}`. Scale:
0 = neutral-template mean projection, 1 = trait-template mean. Cached in
`jobs/{id}/trait/dial.json`. See `docs/api.md`. Do not rebuild it.

D1 extension (what the tree needs to compare the two answers):
- `complete.result.dial` adds `"baseline": [{"text": str, "value": float|null}, ...]`
  and `"steered": [...]` in the same shape: one entry per generated token of
  each pass, with the token's decoded text, so the joined texts equal the
  pass's text. The UI places each value under its word with these texts.
  The baseline pass gets the same `Read` edit (it must not change its tokens).
- Replay mode returns `dial` only if the recording has it.
- One trait per run. Several traits is later (CH).
- Requests without `dial_trait` are unchanged.

## 4. Evidence and patch API [E1]

`GET /api/traits?model_id=...` →
```json
{"traits": [{"id": "<job id>", "family": "hostility", "model_id": "...",
  "version": 2, "verdict": "knob", "layer": 14, "dose": 0.5, "created": 1793300000,
  "cards": {"update_up": {"passed": true, "trait_mean": 0.31, "trait_ci": [0.22, 0.40]},
            "update_down": {...}, "delete": {...}, "prompt_baseline": {...}},
  "guard": {"blocked": false, "would_block": false, "verified": true},
  "patch": "/api/traits/<id>/patch"}]}
```
Only jobs with `status: done`. `patch` is null when no `knob.lobot` exists.

`GET /api/traits/{id}/report` → the job's `report.json` as is.
`GET /api/traits/{id}/report.md` → `text/markdown`.
`GET /api/traits/{id}/patch` → `application/zip`, `Content-Disposition:
attachment; filename="<family>-v<version>.lobot"`.

## 5. File formats

Reader cache `store/reader/<model-slug>/` [R1]:
- `fires_on.npz`: arrays `act` float16 `[L, N, k]` (top activations) and `pos`
  int32 `[L, N, k]` (global token position in the sample), highest first.
- `sample.json`: `{"model_id", "revision", "source": "<data file>",
  "tokens": [str, ...], "doc_starts": [int, ...]}`.
- `cards.json` (optional, for replay): `{"<layer>:<neuron>": <neuron card>}`.

Reader sample text `data/reader_sample.txt` [R1]: at least 50,000 tokens of
public-domain or permissively licensed English, sources and licenses listed in
`data/README.md`. Committed to the repo.

Lift prompts `store/reader/lift_prompts.json` [R1]: 8 neutral prompts that end
right before a free-choice word, for example "My favorite thing to talk about
is". No harmful or political content.

`.lobot` patch [P1 extends, `patch.py` owns]: `patch.json` adds
`"evidence_sha256"` (sha256 of the canonical JSON of `report_card`),
`"members_sha256": {"direction.safetensors": "...", ...}` and
`"lobot_version"`. `load_patch` rejects a mismatch. Old v1 patches without
these fields still load, flagged `"unverified": true`.

Recording with dials (optional) [V1]: a sweep run may carry
`"dial": {...}` in the same shape as `complete.result.dial` (section 3).

## 6. Guard additions [S1]

`/api/state` adds `"guard_default": bool` (env `LOBOT_GUARD_DEFAULT=1` turns it
on; the demo runs with it on). The UI sends `"guard": state.guard_default`
unless the user changes it.
Guard dicts add `"limit": 0.5` and `"floor": 0.2` (the values in force).

## 7. 100X: clean features [F1]

`GET /api/features` → `{"saes": [{"model_id", "layer",
"features", "top_k", "live", "dead", "mse"}]}`. 200 with an empty
list when nothing is trained. No model needed.

`GET /api/feature/{layer}/{index}?k=12` → `{"model_id", "layer",
"index", "live", "pushes_toward": {up, down, "coherence",
"layer_frac"}, "fires_on": [...] | null, "fires_on_cache":
{"built", "tokens"}}`. 400 when the index is out of bounds, 404
when no SAE is trained for the layer, 503 without weights or in
replay mode.

`POST /api/feature-lift` body `{"layer", "index", "dose", "words"
| null, "prompts": null}` → the `steer_lift` dict. `words: null`
uses the feature's top words (top-3 whole-word up tokens, else
top-1). Errors like `/api/lift`, plus 404 without an SAE.

Library: `reader.pushes_of_direction(eng, direction, k, layer)`
(any hidden vector through gamma and the head;
`pushes_toward` shares its tail, same numbers),
`reader.steer_lift_vector(eng, layer, direction, dose, words,
prompts, medians)` (unit-normed direction as a `VectorEdit` with
scale = the layer median norm). `sae.py` owns the TopK SAE,
`store/sae/<model-slug>/L{layer}/` (`sae.safetensors`,
`config.json`, `fires.npz`, `tokens.json`).

## 8. 100X: fast Create [C2]

`POST /api/create` gains `"schedule": "fast"|"full"`
(default full). Status and `report.json` record the schedule,
per-stage `timings`, `early_stop` decisions and base-cache
hits. `trait_eval.run_trait(..., schedule="full")`; full is the
reference pipeline, fast is fewer prompts + shorter generations
+ early stop (preliminary: verdicts disagree, see `reports/C2.md`).

## 9. 100X: open descriptions [C3]

`POST /api/drafts {"description", "seed"}` drafts model-written
parts + 20 candidate replies (side hidden). `GET
/api/drafts/{id}` shows parts + labels so far. `POST
/api/drafts/{id}/labels` merges user labels. `POST
/api/drafts/{id}/validate` registers the family on agreement >=
0.85, else 422 with disagreements. Safety-weakening descriptions
are 422. `TraitFamily` carries persisted `baselines`; run
configs come from the store.

## 10. 100X: chat with live dials [CH]

`POST /api/chat/start` gains `"dial_traits": [...]` (at most
4). Chat `token` events carry `"dial": {trait: value}`;
`done.result` and history carry per-turn dial lists.
`GET /api/chat/{id}/transcript` returns the replay spec;
`POST /api/chat/replay` re-runs it in a fresh session and
reports per-turn match. `Engine.run_stream` takes `dials: [...]`
(`dial` stays as the one-spec shorthand).
