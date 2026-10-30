"""Reader core library (R1): say what a neuron is about, two ways.

Output side ("pushes toward"): project one MLP column through the final
norm and the unembedding, and read the top and bottom tokens.
Input side ("fires on"): cache the top activating contexts per neuron
from a sample, then read the cache without the model.

Contracts: docs/roadmap/contracts.md section 1. All functions run
inside eng.clean() and leave zero hooks behind, also on failure.
"""
from __future__ import annotations

import bisect
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from model_lobotomy import engine
from model_lobotomy import steer

# Cache layout version, written into sample.json. Bump on format change.
CACHE_VERSION = 1
# Default sample file named as the cache source (contracts section 5).
DEFAULT_SOURCE = "data/reader_sample.txt"


# -- small helpers ------------------------------------------------------------

def _require_spec(eng) -> steer.ModelSpec:
    """Spec for the warm model, or a loud error when no model is loaded."""
    if eng is None or eng.model is None or eng.model_id is None:
        raise RuntimeError("no model loaded")
    return eng.spec


def _check_k(k: int, name: str) -> int:
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError(f"{name} must be an int >= 1, got {k!r}")
    if k < 1:
        raise ValueError(f"{name} must be an int >= 1, got {k!r}")
    return k


def _resolve_layers(spec: steer.ModelSpec, layers) -> list[int]:
    if layers is None:
        return list(range(spec.layers))
    out: list[int] = []
    for layer in layers:
        layer = int(layer)
        if not 0 <= layer < spec.layers:
            raise ValueError(
                f"layer {layer} out of bounds [0,{spec.layers})")
        if layer not in out:
            out.append(layer)
    if not out:
        raise ValueError("need at least one layer")
    return out


def _norm_medians(medians: dict) -> dict[int, float]:
    """Medians with int keys (JSON caches store them as strings)."""
    return {int(k): float(v) for k, v in (medians or {}).items()}


def _device_of(model) -> str:
    params = getattr(model, "parameters", None)
    if params is not None:
        first = next(params(), None)
        if first is not None:
            return str(first.device)
    return "cpu"


def _as_id_list(ids) -> list[int]:
    if torch.is_tensor(ids):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids]


def _encode_ids(tok, text: str, add_special: bool) -> torch.Tensor:
    """One text to a 1D id tensor on CPU."""
    out = tok(text, return_tensors="pt", add_special_tokens=add_special)
    try:
        ids = out["input_ids"]
    except (KeyError, TypeError):
        ids = out.input_ids
    if not torch.is_tensor(ids):
        ids = torch.as_tensor(ids, dtype=torch.long)
    return ids.reshape(-1).cpu()


def _word_ids(tok, word: str) -> list[int]:
    out = tok(word, add_special_tokens=False)
    try:
        ids = out["input_ids"]
    except (KeyError, TypeError):
        ids = out.input_ids
    return _as_id_list(ids)


def _decode_one(tok, idx: int) -> str:
    return tok.decode([int(idx)])


def _final_norm_weight(eng) -> torch.Tensor:
    """Final RMSNorm gamma as a float32 CPU vector of width hidden."""
    model = eng.model
    spec = eng.spec
    cands = []
    inner = getattr(model, "model", None)
    if spec.block_path == "qwen":
        lang = getattr(inner, "language_model", None)
        cands.append(getattr(lang, "norm", None))
    cands.append(getattr(inner, "norm", None))
    cands.append(getattr(model, "norm", None))
    for mod in cands:
        w = getattr(mod, "weight", None)
        if w is None:
            continue
        vec = w.detach().float().cpu().reshape(-1)
        if vec.numel() == spec.hidden:
            return vec
    raise RuntimeError(
        "cannot find the final norm weight "
        "(tried model.model.norm and model.model.language_model.norm)")


def _projected_logits(eng, layer: int, neuron: int) -> torch.Tensor:
    """Full-vocab logits of one neuron direction (float32 CPU).

    Logits are lm_head(gamma * W_down[:, neuron]). The single vector
    is scaled by gamma but never re-normalized.
    """
    spec = eng.spec
    down = steer.resolve_down_proj(eng.model, layer, spec)
    gamma = _final_norm_weight(eng)
    head = eng._lm_head()
    if head is None:
        raise RuntimeError("the model has no lm_head")
    with torch.no_grad():
        col = down.weight.detach().float().cpu()[:, neuron].reshape(-1)
        if col.numel() != spec.hidden or gamma.numel() != spec.hidden:
            raise RuntimeError(
                f"width mismatch: column {col.numel()}, gamma "
                f"{gamma.numel()}, hidden {spec.hidden}")
        vec = col * gamma
        mat = head.weight.detach().float().cpu()
        if mat.shape[1] != spec.hidden:
            raise RuntimeError(
                f"head width {mat.shape[1]} != hidden {spec.hidden}")
        logits = mat @ vec
        bias = getattr(head, "bias", None)
        if bias is not None:
            logits = logits + bias.detach().float().cpu().reshape(-1)
    return logits


def _layer_frac(spec: steer.ModelSpec, layer: int) -> float:
    if spec.layers < 2:
        return 0.0
    return layer / (spec.layers - 1)


def _zscore(vec: torch.Tensor) -> torch.Tensor:
    std = float(vec.std(unbiased=False).item())
    if std <= 1e-12:
        return torch.zeros_like(vec)
    return (vec - vec.mean()) / std


def _as_recipe_rows(rows) -> list[steer.RecipeRow]:
    out = []
    for row in (rows or []):
        if isinstance(row, steer.RecipeRow):
            out.append(row)
        elif isinstance(row, dict):
            out.append(steer.RecipeRow(
                int(row["layer"]), int(row["neuron"]), float(row["dose"])))
        else:
            out.append(steer.RecipeRow(
                int(row.layer), int(row.neuron), float(row.dose)))
    return out


# -- contracts section 1 ------------------------------------------------------

def pushes_toward(eng, layer: int, neuron: int, k: int = 12) -> dict:
    """Words one neuron pushes toward (output side).

    Projects W_down[:, neuron] * final_norm.weight through the
    unembedding. Returns up (largest k), down (smallest k), the
    coherence of the up tokens, and the layer fraction.
    """
    spec = _require_spec(eng)
    _check_k(k, "k")
    layer, neuron = int(layer), int(neuron)
    steer.validate_recipe([steer.RecipeRow(layer, neuron, 0.0)], spec)
    with eng.clean():
        logits = _projected_logits(eng, layer, neuron)
    kk = min(k, logits.numel())
    up_vals, up_idx = torch.topk(logits, kk, largest=True, sorted=True)
    down_vals, down_idx = torch.topk(logits, kk, largest=False, sorted=True)
    tok = eng.tokenizer
    up = [{"token": _decode_one(tok, i), "id": int(i), "logit": float(v)}
          for i, v in zip(up_idx.tolist(), up_vals.tolist())]
    down = [{"token": _decode_one(tok, i), "id": int(i), "logit": float(v)}
            for i, v in zip(down_idx.tolist(), down_vals.tolist())]
    return {"up": up, "down": down,
            "coherence": coherence([e["token"] for e in up]),
            "layer_frac": _layer_frac(spec, layer)}


def coherence(tokens: list[str]) -> float:
    """Share of tokens that are whole words, 0..1.

    A token counts when it holds letters only after strip and has
    length >= 3. Each lowercase stem counts once.
    """
    stems: list[str] = []
    seen: set[str] = set()
    for tok in (tokens or []):
        stem = str(tok).strip().lower()
        if stem not in seen:
            seen.add(stem)
            stems.append(stem)
    if not stems:
        return 0.0
    good = sum(1 for s in stems if len(s) >= 3 and s.isalpha())
    return good / len(stems)


def build_fires_on_cache(eng, texts: list[str], out_dir: Path,
                         layers: list[int] | None = None,
                         k: int = 8) -> Path:
    """Build the fires_on cache: top-k activations per neuron.

    Runs each document once with no edits, taps the down_proj input,
    and keeps a running top-k per neuron. Skips position 0 (BOS) of
    each document. Writes out_dir/fires_on.npz (act float16, pos
    int32, both [layers, intermediate, k], highest first) and
    out_dir/sample.json. Returns out_dir.
    """
    spec = _require_spec(eng)
    _check_k(k, "k")
    layers = _resolve_layers(spec, layers)
    docs = [t for t in (texts or []) if t and t.strip()]
    if not docs:
        raise ValueError("build_fires_on_cache needs at least one non-empty text")
    out_dir = Path(out_dir)
    with eng.clean():
        return _build_cache_inner(eng, spec, docs, out_dir, layers, k)


def _mlp_grabber(grabbed: dict, layer: int, width: int):
    def _pre(_mod, args):
        got = args[0].detach().float().cpu()
        if got.shape[-1] != width:
            raise RuntimeError(
                f"capture width mismatch at L{layer}: {got.shape[-1]} != "
                f"intermediate {width}")
        grabbed[layer] = got
    return _pre


def _build_cache_inner(eng, spec, docs, out_dir, layers, k) -> Path:
    tok = eng.tokenizer
    model = eng.model
    device = _device_of(model)
    width = spec.intermediate
    top_act = {L: torch.full((width, k), float("-inf")) for L in layers}
    top_pos = {L: torch.full((width, k), -1, dtype=torch.long) for L in layers}
    tokens: list[str] = []
    doc_starts: list[int] = []
    for doc in docs:
        ids = _encode_ids(tok, doc, add_special=True)
        count = ids.numel()
        if count < 2:
            continue  # no usable position after the BOS skip
        grabbed: dict[int, torch.Tensor] = {}
        handles = []
        for layer in layers:
            mod = steer.resolve_down_proj(model, layer, spec)
            handles.append(mod.register_forward_pre_hook(
                _mlp_grabber(grabbed, layer, width)))
        try:
            with torch.no_grad():
                model(ids.reshape(1, -1).to(device))
        finally:
            for handle in handles:
                handle.remove()
        base = len(tokens)
        tokens.extend(_decode_one(tok, i) for i in ids.tolist())
        doc_starts.append(base)
        pos_row = torch.arange(base + 1, base + count, dtype=torch.long)
        pos_row = pos_row.reshape(1, -1).expand(width, -1)
        for layer in layers:
            mat = grabbed[layer].reshape(-1, width)[1:].transpose(0, 1)
            merged_act = torch.cat([top_act[layer], mat], dim=1)
            merged_pos = torch.cat([top_pos[layer], pos_row], dim=1)
            vals, idx = torch.topk(merged_act, k, dim=1,
                                   largest=True, sorted=True)
            top_act[layer] = vals
            top_pos[layer] = torch.gather(merged_pos, 1, idx)
    if not doc_starts:
        raise ValueError("no document had 2 or more tokens")
    full_act = torch.full((spec.layers, width, k), float("-inf"))
    full_pos = torch.full((spec.layers, width, k), -1, dtype=torch.long)
    for layer in layers:
        full_act[layer] = top_act[layer]
        full_pos[layer] = top_pos[layer]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "fires_on.npz",
             act=full_act.numpy().astype(np.float16),
             pos=full_pos.numpy().astype(np.int32))
    (out_dir / "sample.json").write_text(json.dumps({
        "model_id": eng.model_id, "revision": eng.revision,
        "source": DEFAULT_SOURCE, "cache_version": CACHE_VERSION,
        "tokens": tokens, "doc_starts": doc_starts,
    }))
    return out_dir


def fires_on(cache_dir: Path, layer: int, neuron: int, k: int = 8,
             window: int = 6) -> list[dict]:
    """Top cached contexts for one neuron. Reads the cache only.

    Returns up to k entries, highest activation first. Each entry
    carries the activation, the token, a +/- window decoded context
    clamped to its document, and the char span of the token in it.
    """
    _check_k(k, "k")
    window = int(window)
    if window < 0:
        raise ValueError(f"window must be >= 0, got {window}")
    cache_dir = Path(cache_dir)
    with np.load(cache_dir / "fires_on.npz") as data:
        act = data["act"]
        pos = data["pos"]
    count_l, count_n, cached_k = act.shape
    layer, neuron = int(layer), int(neuron)
    if not 0 <= layer < count_l or not 0 <= neuron < count_n:
        raise ValueError(
            f"neuron L{layer}N{neuron} out of the cached bounds "
            f"[0,{count_l})x[0,{count_n})")
    sample = json.loads((cache_dir / "sample.json").read_text())
    tokens = sample["tokens"]
    starts = [int(s) for s in sample["doc_starts"]]
    out: list[dict] = []
    for i in range(min(k, cached_k)):
        at = int(pos[layer, neuron, i])
        if at < 0 or at >= len(tokens):
            continue
        doc = bisect.bisect_right(starts, at) - 1
        doc_start = starts[doc]
        doc_end = starts[doc + 1] if doc + 1 < len(starts) else len(tokens)
        lo = max(doc_start, at - window)
        hi = min(doc_end, at + window + 1)
        token = tokens[at]
        start = sum(len(t) for t in tokens[lo:at])
        out.append({"activation": float(act[layer, neuron, i]),
                    "token": token,
                    "context": "".join(tokens[lo:hi]),
                    "mark": [start, start + len(token)]})
    return out


def find_neurons(eng, word: str, layers: list[int] | None = None,
                 k: int = 20) -> dict:
    """Neurons that push one word hardest.

    Score is the logit of the word's first token in the projected
    neuron direction, z-scored within its layer. Rank is the 1-based
    position of that token in the neuron's own up list, or null when
    past 100.
    """
    spec = _require_spec(eng)
    _check_k(k, "k")
    if not word or not word.strip():
        raise ValueError("find_neurons needs a non-empty word")
    token_ids = _word_ids(eng.tokenizer, word)
    if not token_ids:
        raise ValueError(f"word {word!r} encodes to no tokens")
    layers = _resolve_layers(spec, layers)
    first = token_ids[0]
    with eng.clean():
        gamma = _final_norm_weight(eng)
        head = eng._lm_head()
        if head is None:
            raise RuntimeError("the model has no lm_head")
        with torch.no_grad():
            rows = head.weight.detach().float().cpu()
            if first >= rows.shape[0]:
                raise ValueError(
                    f"token id {first} is outside the head "
                    f"(vocab {rows.shape[0]})")
            vec = gamma * rows[first]
            scored: list[tuple[float, int, int]] = []
            for layer in layers:
                down = steer.resolve_down_proj(eng.model, layer, spec)
                mat = down.weight.detach().float().cpu()
                z = _zscore(vec @ mat)
                for neuron in range(z.numel()):
                    scored.append((float(z[neuron].item()), layer, neuron))
        scored.sort(key=lambda row: -row[0])
        results = []
        for score, layer, neuron in scored[:k]:
            logits = _projected_logits(eng, layer, neuron)
            top = min(100, logits.numel())
            _, idx = torch.topk(logits, top, largest=True, sorted=True)
            ids = idx.tolist()
            rank = ids.index(first) + 1 if first in ids else None
            results.append({"layer": layer, "neuron": neuron,
                            "score": score, "rank": rank})
    return {"word": word, "token_ids": token_ids, "results": results}


def word_neurons(eng, prompt: str, rows: list, run: str, token_index: int,
                 medians: dict, k: int = 10) -> dict:
    """Neurons most active at one prompt token.

    Re-runs the wrapped prompt once (clean for "base", with rows for
    "steered"), taps every down_proj input at token_index, and ranks
    neurons by z-score from the same pass (per-layer mean and std at
    that position). token_index is 1-based into the wrapped prompt
    tokens (0 is BOS and has no row); it must address the prompt.
    """
    spec = _require_spec(eng)
    _check_k(k, "k")
    if run not in ("base", "steered"):
        raise ValueError(f'run must be "base" or "steered", got {run!r}')
    if not prompt or not prompt.strip():
        raise ValueError("word_neurons needs a non-empty prompt")
    if run == "steered":
        recipe = _as_recipe_rows(rows)
        steer.validate_recipe(recipe, spec)
        edits = engine.EditSet.from_rows(recipe)
    else:
        edits = engine.EMPTY_EDITS
    meds = _norm_medians(medians)
    with eng.clean():
        wrapped = steer.wrap_prompt_notk(eng.tokenizer, prompt, spec)
        ids = _encode_ids(eng.tokenizer, wrapped, add_special=True)
        total = ids.numel()
        token_index = int(token_index)
        if not 1 <= token_index < total:
            raise ValueError(
                f"token_index {token_index} out of bounds [1,{total}) "
                f"for the wrapped prompt ({total} tokens)")
        activ = eng.capture_with_edits(
            [prompt], list(range(spec.layers)), edits, meds,
            point="mlp_in", reduce="none")
        row = token_index - 1
        scored: list[tuple[float, float, int, int]] = []
        for layer in range(spec.layers):
            vec = activ[layer][0].float()[row]
            z = _zscore(vec)
            for neuron in range(vec.numel()):
                scored.append((float(z[neuron].item()),
                              float(vec[neuron].item()), layer, neuron))
    scored.sort(key=lambda entry: -entry[0])
    results = [{"layer": layer, "neuron": neuron,
                "activation": act, "z": z}
               for z, act, layer, neuron in scored[:k]]
    return {"token": _decode_one(eng.tokenizer, int(ids[token_index].item())),
            "token_index": token_index, "results": results}


def steer_lift(eng, layer: int, neuron: int, dose: float, words: list[str],
               prompts: list[str], medians: dict) -> dict:
    """How much steering one neuron lifts its predicted words.

    Reads the next-token probability of each word's first token over
    each prompt, with and without a NeuronEdit at dose. base_p and
    steered_p average over prompts per word; lift is the geometric
    mean of steered_p / base_p over words and prompts.
    """
    spec = _require_spec(eng)
    layer, neuron = int(layer), int(neuron)
    steer.validate_recipe([steer.RecipeRow(layer, neuron, 0.0)], spec)
    try:
        dose = float(dose)
    except (TypeError, ValueError):
        raise ValueError(f"dose must be a number, got {dose!r}") from None
    if not math.isfinite(dose):
        raise ValueError(f"dose must be finite, got {dose!r}")
    words = list(words or [])
    prompts = list(prompts or [])
    if not words:
        raise ValueError("steer_lift needs at least one word")
    if not prompts:
        raise ValueError("steer_lift needs at least one prompt")
    if any(not w or not str(w).strip() for w in words):
        raise ValueError("steer_lift words must be non-empty")
    if any(not p or not str(p).strip() for p in prompts):
        raise ValueError("steer_lift prompts must be non-empty")
    first_ids = []
    for word in words:
        ids = _word_ids(eng.tokenizer, word)
        if not ids:
            raise ValueError(f"word {word!r} encodes to no tokens")
        first_ids.append(ids[0])
    meds = _norm_medians(medians)
    edits = engine.EditSet(
        neuron=(engine.NeuronEdit(layer, neuron, dose),))
    base_acc = [0.0] * len(words)
    steered_acc = [0.0] * len(words)
    log_ratios: list[float] = []
    with eng.clean():
        for prompt in prompts:
            base = torch.softmax(
                eng.prefill(prompt).float().cpu(), dim=-1)
            steered = torch.softmax(
                eng.prefill(prompt, edits, medians=meds).float().cpu(),
                dim=-1)
            for j, tok_id in enumerate(first_ids):
                base_p = max(float(base[tok_id].item()), 1e-12)
                steered_p = max(float(steered[tok_id].item()), 1e-12)
                base_acc[j] += base_p
                steered_acc[j] += steered_p
                log_ratios.append(math.log(steered_p / base_p))
    count = len(prompts)
    base_mean = [v / count for v in base_acc]
    steered_mean = [v / count for v in steered_acc]
    lift = float(math.exp(sum(log_ratios) / len(log_ratios)))
    return {"words": words, "base_p": base_mean,
            "steered_p": steered_mean, "lift": lift}


def main(argv: list[str] | None = None) -> int:
    """Print one neuron card (layer, neuron, up, fires_on) as text."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Print one reader neuron card from the warm model.")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--neuron", type=int, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--cache", default=None,
                        help="fires_on cache dir (else no fires_on lines)")
    args = parser.parse_args(argv)
    eng = engine.get_engine()
    if eng.model is None:
        if not args.model:
            print("no model loaded; pass --model <id>", file=sys.stderr)
            return 2
        eng.load(args.model)
    card = pushes_toward(eng, args.layer, args.neuron)
    print(f"L{args.layer}N{args.neuron} "
          f"coherence={card['coherence']:.2f} "
          f"layer_frac={card['layer_frac']:.2f}")
    print("up: " + ", ".join(
        f"{e['token']} ({e['logit']:.2f})" for e in card["up"]))
    print("down: " + ", ".join(
        f"{e['token']} ({e['logit']:.2f})" for e in card["down"]))
    if args.cache:
        for entry in fires_on(args.cache, args.layer, args.neuron, k=3):
            print(f"{entry['activation']:.2f} {entry['token']!r}: "
                  f"{entry['context']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
