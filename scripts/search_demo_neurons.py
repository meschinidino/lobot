"""Search demo neurons on one model (R3).

Finds neurons that are both readable (coherent up words, matching
fires_on contexts) and steerable (lift >= 3 with a coherent steered
answer). Needs real weights: book docs/roadmap/model-lane.md first.

Stages:
  1. Coherence scan over --layers (batched matmuls, exact top-k per
     neuron); keep --top by coherence.
  2. Add find_neurons hits for demo-friendly words.
  3. Score the pool: exact pushes_toward, fires_on agreement.
  4. Lift screen at dose +2 for the top --screen by (coherence,
     agreement). Words need base_p >= 1e-4 to count.
  5. Survivors: lift at +1/+3, demo generation, coherent() check.
  6. Rank, write logs/reader/<slug>/shortlist.json and
     store/reader/<slug>/cards.json (top --cards).

Example:
  uv run python scripts/search_demo_neurons.py \\
      --model meta-llama/Llama-3.2-3B-Instruct --layers 12-27 --top 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from model_lobotomy import engine as engine_mod
from model_lobotomy import reader
from model_lobotomy import steer
from derive_llama_recipe import coherent

DEMO_WORDS = ["pirate", "ocean", "music", "food", "space", "weather",
              "poetry", "money", "sports", "animal"]
DEMO_PROMPT = "Tell me about your weekend."
FIND_K = 20
SCAN_K = 12
FIRES_K = 8
COHERENCE_BAR = 0.5
AGREE_BAR = 0.5
LIFT_BAR = 3.0
# Floor on steered_p, not base_p: content words are rare next-tokens on
# neutral prompts (typical base_p 1e-7..1e-9), so a base_p floor admits
# nothing. The steered_p floor keeps the anti-noise intent: the pushed
# word must become genuinely likely.
STEERED_P_FLOOR = 1e-4


def slug_for(model_id: str) -> str:
    return model_id.lower().replace("/", "-").replace("_", "-")


def parse_layer_range(text: str, count: int) -> list[int]:
    lo_s, hi_s = text.split("-", 1)
    layers = list(range(int(lo_s), int(hi_s) + 1))
    for layer in layers:
        if not 0 <= layer < count:
            raise ValueError(f"layer {layer} out of bounds [0,{count})")
    if not layers:
        raise ValueError(f"bad layer range {text!r}")
    return layers


def whole_word(token: str) -> bool:
    stem = str(token).strip()
    return len(stem) >= 3 and stem.isalpha()


def agreement(fires_tokens: list[str], up_tokens: list[str]) -> float:
    """Share of fires_on tokens whose stem sits in the up stems.

    No outside synonym list exists in the repo, so this is an exact
    lowercase-stem match against the neuron's own up list.
    """
    up_stems = {str(t).strip().lower() for t in up_tokens}
    stems = [str(t).strip().lower() for t in (fires_tokens or [])]
    if not stems:
        return 0.0
    return sum(1 for s in stems if s in up_stems) / len(stems)


def lift_words(up_tokens: list[str], count: int = 3) -> list[str]:
    """First whole-word up tokens (raw strings), filled from the rest."""
    words = [t for t in up_tokens if whole_word(t)][:count]
    for tok in up_tokens:
        if len(words) >= count:
            break
        if tok not in words:
            words.append(tok)
    return words[:count]


def rank_candidates(cands: list[dict]) -> list[dict]:
    """Rank by: passes both checks, then lift, then coherence.

    Pure (no model): readable means coherence and agreement at or
    above bar; steerable means best lift at or above bar with a
    coherent steered answer. Mutates nothing; returns new dicts with
    a "passes" flag, sorted best first.
    """
    out = []
    for cand in cands:
        lifts = cand.get("lifts") or {}
        best = max((v["lift"] for v in lifts.values()
                    if v.get("lift") is not None), default=None)
        # Readable means coherent up words; agreement is reported and
        # breaks ties, but exact-stem matching undercounts real
        # agreement (fires "rained"/"drizzly" vs up "rain"), so it
        # does not gate. The winner's contexts get a human read.
        readable = (cand.get("coherence") or 0.0) >= COHERENCE_BAR
        steerable = (best is not None and best >= LIFT_BAR
                     and cand.get("coherent_text") is True)
        row = dict(cand)
        row["passes"] = bool(readable and steerable)
        row["best_lift"] = best
        out.append(row)
    out.sort(key=lambda r: (
        not r["passes"],
        -(r["best_lift"] if r["best_lift"] is not None else -1.0),
        -(r.get("coherence") or 0.0),
        r.get("layer", 0), r.get("neuron", 0)))
    return out


def shortlist_entry(layer: int, neuron: int) -> dict:
    """Blank scored-candidate record (keys fixed for shortlist.json)."""
    return {"layer": layer, "neuron": neuron, "coherence": None,
            "agreement": None, "up": [], "fires_sample": [],
            "lift_words": [], "lifts": {}, "best_dose": None,
            "dose_texts": {}, "steered_text": None,
            "coherent_text": None}


def scan_layer(eng, layer: int, top_k: int = SCAN_K,
               batch: int = 512) -> list[tuple[float, int, list]]:
    """(coherence, neuron, up_ids) for every neuron in one layer.

    Batched: one [vocab, batch] matmul per batch on the model
    device, exact top-k per neuron, each unique id decoded once.
    """
    spec = eng.spec
    model = eng.model
    tok = eng.tokenizer
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = "cpu"
    down = steer.resolve_down_proj(model, layer, spec)
    gamma = reader._final_norm_weight(eng)
    head = eng._lm_head()
    if head is None:
        raise RuntimeError("the model has no lm_head")
    with torch.no_grad():
        mat = down.weight.detach().float().cpu()
        vecs = mat * gamma.reshape(-1, 1)
        rows = head.weight.detach().float()
        if getattr(head, "bias", None) is not None:
            bias = head.bias.detach().float()
        else:
            bias = None
        top_ids: list[list[int]] = [[] for _ in range(spec.intermediate)]
        for start in range(0, spec.intermediate, batch):
            sl = slice(start, start + batch)
            logits = (rows.to(device) @ vecs[:, sl].to(device)).float()
            if bias is not None:
                logits = logits + bias.reshape(-1, 1).to(logits.device)
            _, idx = torch.topk(logits, top_k, dim=0, largest=True,
                                sorted=True)
            idx = idx.cpu()
            for j in range(idx.shape[1]):
                top_ids[start + j] = idx[:, j].tolist()
            del logits, idx
    uniq = sorted({i for ids in top_ids for i in ids})
    words = {i: tok.decode([i]) for i in uniq}
    out = []
    for neuron, ids in enumerate(top_ids):
        ups = [words[i] for i in ids]
        out.append((reader.coherence(ups), neuron, ups))
    return out


def lift_gated(eng, layer: int, neuron: int, dose: float,
               words: list[str], prompts: list[str],
               medians: dict) -> dict | None:
    """steer_lift over words that clear the steered_p floor.

    Drops words the push never makes likely (steered_p under
    STEERED_P_FLOOR) and re-runs once when the set shrinks.
    Returns None when no word counts.
    """
    got = reader.steer_lift(eng, layer, neuron, dose, words, prompts,
                            medians)
    keep = [w for w, p in zip(got["words"], got["steered_p"])
            if p >= STEERED_P_FLOOR]
    if not keep:
        return None
    if len(keep) < len(words):
        got = reader.steer_lift(eng, layer, neuron, dose, keep,
                                prompts, medians)
    return got


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Search readable, steerable demo neurons.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", default="12-27")
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--screen", type=int, default=0,
                        help="lift-screen the top N by (coherence, "
                             "agreement); 0 screens the whole pool")
    parser.add_argument("--cards", type=int, default=10)
    parser.add_argument("--words", default=",".join(DEMO_WORDS))
    parser.add_argument("--cache", default=None,
                        help="fires_on cache dir (default: store path)")
    parser.add_argument("--out", default=None,
                        help="shortlist path (default: logs path)")
    args = parser.parse_args(argv)
    t_all = time.perf_counter()
    peak_mib = 0.0

    def peak():
        nonlocal peak_mib
        mib = steer.mps_current_mib()
        if mib:
            peak_mib = max(peak_mib, mib)

    eng = engine_mod.get_engine()
    print(f"loading {args.model} ...", flush=True)
    eng.load(args.model)
    try:
        return _search(eng, args, t_all, peak,
                        lambda: peak_mib)
    finally:
        eng.unload()


def _search(eng, args, t_all, peak, get_peak) -> int:
    spec = eng.spec
    slug = slug_for(eng.model_id)
    layers = parse_layer_range(args.layers, spec.layers)
    cache_dir = Path(args.cache or str(ROOT / "store" / "reader" / slug))
    out_path = Path(args.out or
                    str(ROOT / "logs" / "reader" / slug / "shortlist.json"))
    have_cache = ((cache_dir / "fires_on.npz").exists()
                  and (cache_dir / "sample.json").exists())
    if not have_cache:
        print("WARNING: no fires_on cache; agreement stays null "
              "(build it with scripts/build_reader_cache.py).", flush=True)
    lift_prompts = json.loads(
        (ROOT / "store" / "reader" / "lift_prompts.json").read_text())

    print(f"stage 1: coherence scan over {len(layers)} layers ...",
          flush=True)
    pool: dict[tuple[int, int], dict] = {}
    scored = 0
    for layer in layers:
        for coh, neuron, _ in scan_layer(eng, layer):
            entry = shortlist_entry(layer, neuron)
            entry["coherence"] = coh
            pool[(layer, neuron)] = entry
            scored += 1
        peak()
    top = sorted(pool.values(), key=lambda e: (
        -(e["coherence"] or 0.0), e["layer"], e["neuron"]))[:args.top]
    keep = {(e["layer"], e["neuron"]) for e in top}
    print(f"stage 1: kept {len(keep)} of {scored} "
          f"(best coherence {top[0]['coherence']:.2f})", flush=True)

    print("stage 2: find adds ...", flush=True)
    for word in [w for w in args.words.split(",") if w.strip()]:
        try:
            found = reader.find_neurons(eng, word.strip(), layers,
                                        k=FIND_K)
        except ValueError as e:
            print(f"  {word}: skip ({e})", flush=True)
            continue
        for res in found["results"]:
            key = (res["layer"], res["neuron"])
            if key not in keep:
                keep.add(key)
                if key not in pool:
                    entry = shortlist_entry(*key)
                    entry["coherence"] = None
                    pool[key] = entry
        peak()
    print(f"stage 2: pool is {len(keep)} neurons", flush=True)

    print("stage 3: exact cards and agreement ...", flush=True)
    for i, key in enumerate(sorted(keep)):
        entry = pool[key]
        pushes = reader.pushes_toward(eng, key[0], key[1], k=SCAN_K)
        entry["coherence"] = pushes["coherence"]
        entry["up"] = [e["token"] for e in pushes["up"]]
        if have_cache:
            fires = reader.fires_on(cache_dir, key[0], key[1],
                                    k=FIRES_K)
            entry["fires_sample"] = [
                {"token": f["token"], "activation": f["activation"],
                 "context": f["context"]} for f in fires]
            entry["agreement"] = agreement(
                [f["token"] for f in fires], entry["up"])
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(keep)} ...", flush=True)
        peak()

    ordered = sorted(
        (pool[k] for k in keep),
        key=lambda e: (-(e["coherence"] or 0.0),
                       -((e["agreement"] or 0.0)
                         if e["agreement"] is not None else -1.0),
                       e["layer"], e["neuron"]))
    cands = ordered if args.screen <= 0 else ordered[:args.screen]
    print(f"stage 4: lift screen at +2 for {len(cands)} ...", flush=True)
    medians = None
    for i, entry in enumerate(cands):
        words = lift_words(entry["up"])
        entry["lift_words"] = words
        if medians is None:
            _m, _ = steer.calibrate_layer_medians(
                eng.model, eng.tokenizer, list(lift_prompts), layers,
                cache_path=ROOT / steer.calibration_path_for(
                    eng.model_id),
                key=steer.calibration_cache_key(
                    eng.model_id, eng.revision, eng.tokenizer,
                    str(steer.DTYPE), str(steer.DEVICE)),
                spec=spec)
            medians = {int(k): float(v) for k, v in _m.items()}
        got = lift_gated(eng, entry["layer"], entry["neuron"], 2.0,
                         words, lift_prompts, medians)
        if got is not None:
            entry["lifts"]["+2"] = {
                "words": got["words"], "base_p": got["base_p"],
                "steered_p": got["steered_p"], "lift": got["lift"]}
        got2 = entry["lifts"].get("+2", {}).get("lift")
        agr = entry["agreement"]
        print(f"  L{entry['layer']}N{entry['neuron']}: "
              f"coh={entry['coherence']:.2f} "
              f"agr={'n/a' if agr is None else f'{agr:.2f}'} "
              f"lift={'n/a' if got2 is None else f'{got2:.1f}'} "
              f"up={entry['up'][:5]}", flush=True)
        peak()

    survivors = [e for e in cands
                 if (e["lifts"].get("+2", {}).get("lift") or 0.0) >= LIFT_BAR]
    print(f"stage 5: {len(survivors)} survivors at lift >= {LIFT_BAR}; "
          f"full doses and demo runs ...", flush=True)
    from model_lobotomy.engine import EditSet, NeuronEdit
    for entry in survivors:
        words = entry["lift_words"]
        for dose in (1.0, 3.0):
            got = lift_gated(eng, entry["layer"], entry["neuron"],
                             dose, words, lift_prompts, medians)
            if got is not None:
                entry["lifts"][f"+{dose:g}"] = {
                    "words": got["words"], "base_p": got["base_p"],
                    "steered_p": got["steered_p"], "lift": got["lift"]}
        passing = sorted(
            float(d[1:]) for d, v in entry["lifts"].items()
            if (v.get("lift") or 0.0) >= LIFT_BAR)
        for dose in passing:
            edits = EditSet(neuron=(NeuronEdit(
                entry["layer"], entry["neuron"], dose),))
            gen = eng.run(DEMO_PROMPT, edits, medians=medians,
                          max_new_tokens=100, seed=0)
            ok = coherent(gen["text"])
            entry["dose_texts"][f"+{dose:g}"] = {
                "text": gen["text"], "coherent": ok}
            print(f"  L{entry['layer']}N{entry['neuron']} "
                  f"dose={dose}: coherent={ok} "
                  f"text={gen['text'][:100]!r}", flush=True)
        best = next((d for d in passing
                     if entry["dose_texts"][f"+{d:g}"]["coherent"]),
                    passing[0] if passing else None)
        entry["best_dose"] = best
        if best is not None:
            entry["steered_text"] = (
                entry["dose_texts"][f"+{best:g}"]["text"])
            entry["coherent_text"] = (
                entry["dose_texts"][f"+{best:g}"]["coherent"])
        peak()

    ranked = rank_candidates([pool[k] for k in keep])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model_id": eng.model_id, "revision": eng.revision,
        "layers": layers, "demo_prompt": DEMO_PROMPT,
        "candidates": ranked,
    }, indent=2))
    cards_dir = ROOT / "store" / "reader" / slug
    cards_dir.mkdir(parents=True, exist_ok=True)
    cards = {}
    tokens = 0
    if have_cache:
        sample = json.loads((cache_dir / "sample.json").read_text())
        tokens = len(sample["tokens"])
    for entry in ranked[:args.cards]:
        key = (entry["layer"], entry["neuron"])
        pushes = reader.pushes_toward(eng, key[0], key[1], k=SCAN_K)
        fires = (reader.fires_on(cache_dir, key[0], key[1], k=FIRES_K)
                 if have_cache else None)
        cards[f"{key[0]}:{key[1]}"] = {
            "model_id": eng.model_id, "layer": key[0],
            "neuron": key[1], "pushes_toward": pushes,
            "fires_on": fires,
            "fires_on_cache": {"built": have_cache, "tokens": tokens}}
    (cards_dir / "cards.json").write_text(json.dumps(cards, indent=2))
    total_s = time.perf_counter() - t_all
    passed = [e for e in ranked if e["passes"]]
    print(f"wrote {out_path} ({len(ranked)} candidates) and "
          f"{cards_dir / 'cards.json'} ({len(cards)} cards)", flush=True)
    print(f"passing both checks: {len(passed)}; "
          f"model time {total_s / 60:.1f} min; "
          f"peak {get_peak():.0f} MiB", flush=True)
    for entry in ranked[:5]:
        print(f"  L{entry['layer']}N{entry['neuron']}: "
              f"passes={entry['passes']} best_lift={entry['best_lift']} "
              f"coh={entry['coherence']} up={entry['up'][:6]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
