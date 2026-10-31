"""Clean features (F1): a TopK sparse autoencoder per layer.

Trains on cached residual-stream (block_out) vectors, one SAE per
chosen layer. A feature is one decoder row (unit norm): it reads
through reader.pushes_of_direction and steers as an engine.VectorEdit
with scale = the layer median norm.

Store layout store/sae/<model-slug>/L{layer}/: sae.safetensors
(W_enc, b_enc, W_dec, b_dec), config.json (shape + train stats),
fires.npz (top-k encoder acts per feature), tokens.json (the sample
the fires point into). Weights and caches are NOT committed;
config.json files are (see .gitignore).
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import torch

from model_lobotomy import engine as engine_mod
from model_lobotomy import reader
from model_lobotomy import steer

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "sae.safetensors"
FIRES_NAME = "fires.npz"
TOKENS_NAME = "tokens.json"

# Shipped train setting (scripts/train_sae.py defaults match).
EXPANSION = 8
TOP_K = 32
EPOCHS = 8
BATCH = 4096
LR = 3e-4
TRAIN_SEED = 0


class TopkSAE(torch.nn.Module):
    """TopK SAE: exactly top_k active features per row.

    W_enc [hidden, features], W_dec [features, hidden] with unit-norm
    rows (feature directions), b_enc [features], b_dec [hidden].
    """

    def __init__(self, hidden: int, features: int, top_k: int,
                 seed: int = 0):
        super().__init__()
        if hidden < 1 or features < 1:
            raise ValueError(
                f"hidden and features must be >= 1, got {hidden}, {features}")
        if not 1 <= top_k <= features:
            raise ValueError(
                f"top_k must be in [1, {features}], got {top_k}")
        self.hidden = int(hidden)
        self.features = int(features)
        self.top_k = int(top_k)
        gen = torch.Generator().manual_seed(seed)
        self.W_enc = torch.nn.Parameter(torch.randn(
            hidden, features, generator=gen) * (1.0 / hidden) ** 0.5)
        self.b_enc = torch.nn.Parameter(torch.zeros(features))
        dec = torch.randn(features, hidden, generator=gen)
        dec = dec / dec.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.W_dec = torch.nn.Parameter(dec)
        self.b_dec = torch.nn.Parameter(torch.zeros(hidden))

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.data /= self.W_dec.data.norm(
            dim=1, keepdim=True).clamp_min(1e-12)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.W_enc + self.b_enc
        vals, idx = torch.topk(pre, self.top_k, dim=-1)
        codes = torch.zeros_like(pre)
        codes.scatter_(-1, idx, vals)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return codes @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        codes = self.encode(x)
        return self.decode(codes), codes


def feature_direction(sae: TopkSAE, index: int) -> torch.Tensor:
    """Decoder row i as a unit-norm CPU vector (the feature direction)."""
    index = int(index)
    if not 0 <= index < sae.features:
        raise ValueError(
            f"feature {index} out of bounds [0,{sae.features})")
    with torch.no_grad():
        vec = sae.W_dec.detach().float().cpu()[index]
        return vec / vec.norm().clamp_min(1e-12)


def train_sae(acts: torch.Tensor, *, hidden: int, features: int,
              top_k: int = TOP_K, epochs: int = EPOCHS,
              batch: int = BATCH, lr: float = LR, seed: int = 0,
              device: str = "cpu", log=None) -> tuple:
    """Train a TopK SAE on cached [N, hidden] acts. Returns (sae, stats).

    Loss is plain MSE; decoder rows are re-normalized after each step.
    stats: {mse_first, mse_last, dead, live, l0_mean, steps, seed}.
    Dead = never active over the full acts (measured after training).
    """
    acts = torch.as_tensor(acts, dtype=torch.float32)
    if acts.dim() != 2 or acts.shape[1] != hidden:
        raise ValueError(
            f"acts must be [N, {hidden}], got {list(acts.shape)}")
    if acts.shape[0] < 1:
        raise ValueError("train_sae needs at least one activation row")
    sae = TopkSAE(hidden, features, top_k, seed=seed).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    n = acts.shape[0]
    first, last, steps = None, None, 0
    sae.train()
    for _ in range(max(int(epochs), 1)):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, max(int(batch), 1)):
            rows = perm[s:s + max(int(batch), 1)]
            x = acts[rows].to(device)
            recon, _ = sae(x)
            loss = torch.nn.functional.mse_loss(recon, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                sae.normalize_decoder()
            val = float(loss.item())
            if first is None:
                first = val
            last = val
            steps += 1
            if log is not None:
                log(steps, val)
    sae.eval()
    counts = torch.zeros(features, dtype=torch.long)
    with torch.no_grad():
        for s in range(0, n, max(int(batch), 1)):
            x = acts[s:s + max(int(batch), 1)].to(device)
            counts += (sae.encode(x).cpu() != 0).sum(dim=0)
    live = int((counts > 0).sum().item())
    stats = {"mse_first": first, "mse_last": last, "live": live,
             "dead": features - live, "l0_mean": float(top_k),
             "steps": steps, "seed": seed}
    return sae.cpu(), stats


def save_sae(out_dir, sae: TopkSAE, config: dict) -> Path:
    """Write sae.safetensors + config.json into out_dir. Returns out_dir."""
    from safetensors.torch import save_file
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"hidden": sae.hidden, "features": sae.features,
           "top_k": sae.top_k, **dict(config)}
    save_file({"W_enc": sae.W_enc.detach().cpu(),
               "b_enc": sae.b_enc.detach().cpu(),
               "W_dec": sae.W_dec.detach().cpu(),
               "b_dec": sae.b_dec.detach().cpu()},
              out_dir / WEIGHTS_NAME)
    (out_dir / CONFIG_NAME).write_text(json.dumps(cfg, indent=2))
    return out_dir


def load_sae(sae_dir) -> tuple:
    """Load (sae, config) from a save_sae dir."""
    from safetensors.torch import load_file
    sae_dir = Path(sae_dir)
    cfg = json.loads((sae_dir / CONFIG_NAME).read_text())
    tensors = load_file(sae_dir / WEIGHTS_NAME)
    sae = TopkSAE(int(cfg["hidden"]), int(cfg["features"]),
                  int(cfg["top_k"]))
    with torch.no_grad():
        sae.W_enc.copy_(tensors["W_enc"])
        sae.b_enc.copy_(tensors["b_enc"])
        sae.W_dec.copy_(tensors["W_dec"])
        sae.b_dec.copy_(tensors["b_dec"])
    return sae.eval(), cfg


def sae_dir_for(store_root, model_id: str, layer: int) -> Path:
    slug = model_id.lower().replace("/", "-").replace("_", "-")
    return Path(store_root) / "sae" / slug / f"L{int(layer)}"


def cache_block_acts(eng, texts: list[str], layer: int) -> dict:
    """Per-token block_out vectors for texts: {acts, tokens, doc_starts}.

    acts is [N, hidden] float32 CPU (position 0/BOS of each document
    skipped, like the R1 cache). Runs inside eng.clean().
    """
    if eng is None or eng.model is None or eng.model_id is None:
        raise RuntimeError("no model loaded")
    spec = eng.spec
    layer = int(layer)
    if not 0 <= layer < spec.layers:
        raise ValueError(
            f"layer {layer} out of bounds [0,{spec.layers})")
    docs = [t for t in (texts or []) if t and t.strip()]
    if not docs:
        raise ValueError("cache_block_acts needs at least one non-empty text")
    with eng.clean():
        return _cache_inner(eng, spec, docs, layer)


def _cache_inner(eng, spec, docs, layer) -> dict:
    tok = eng.tokenizer
    model = eng.model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = "cpu"
    grabbed: dict = {}
    block = steer.resolve_block(model, layer, spec)

    def _grab(_m, _a, out):
        # Must return None: any other return replaces the block output.
        grabbed["h"] = steer._block_hidden(out).detach().float().cpu()

    handle = block.register_forward_hook(_grab)
    acts, tokens, doc_starts = [], [], []
    try:
        with torch.no_grad():
            for doc in docs:
                out = tok(doc, return_tensors="pt", add_special_tokens=True)
                try:
                    ids = out["input_ids"]
                except (KeyError, TypeError):
                    ids = out.input_ids
                if not torch.is_tensor(ids):
                    ids = torch.as_tensor(ids, dtype=torch.long)
                ids = ids.reshape(1, -1)
                if ids.numel() < 2:
                    continue
                grabbed.clear()
                model(ids.to(device))
                rows = grabbed["h"].reshape(-1, spec.hidden)[1:]
                if rows.shape[-1] != spec.hidden:
                    raise RuntimeError(
                        f"capture width mismatch at L{layer}: "
                        f"{rows.shape[-1]} != hidden {spec.hidden}")
                base = len(tokens)
                tokens.extend(tok.decode([int(i)])
                              for i in ids.reshape(-1).tolist())
                doc_starts.append(base)
                acts.append(rows)
    finally:
        handle.remove()
    if not doc_starts:
        raise ValueError("no document had 2 or more tokens")
    return {"acts": torch.cat(acts, dim=0),
            "tokens": tokens, "doc_starts": doc_starts}


def build_feature_fires(sae: TopkSAE, acts: torch.Tensor,
                        tokens: list[str], doc_starts: list[int],
                        out_dir, k: int = 8, batch: int = 2048) -> Path:
    """Top-k encoder acts per feature -> out_dir/fires.npz + tokens.json.

    act [F, k] float16, pos [F, k] int32 (global token position),
    highest first. Encodes in batches; never holds [N, F].
    """
    sae = sae.eval()
    feats = int(sae.features)
    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    acts = torch.as_tensor(acts, dtype=torch.float32)
    n = acts.shape[0]
    if n < 1:
        raise ValueError("build_feature_fires needs at least one row")
    top_act = torch.full((feats, k), float("-inf"))
    top_pos = torch.full((feats, k), -1, dtype=torch.long)
    with torch.no_grad():
        for s in range(0, n, max(int(batch), 1)):
            mat = sae.encode(acts[s:s + max(int(batch), 1)]).transpose(0, 1)
            pos_row = torch.arange(s, s + mat.shape[1], dtype=torch.long)
            pos_row = pos_row.reshape(1, -1).expand(feats, -1)
            merged_act = torch.cat([top_act, mat], dim=1)
            merged_pos = torch.cat([top_pos, pos_row], dim=1)
            vals, idx = torch.topk(merged_act, k, dim=1,
                                   largest=True, sorted=True)
            top_act = vals
            top_pos = torch.gather(merged_pos, 1, idx)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / FIRES_NAME,
             act=top_act.numpy().astype(np.float16),
             pos=top_pos.numpy().astype(np.int32))
    (out_dir / TOKENS_NAME).write_text(json.dumps({
        "tokens": list(tokens),
        "doc_starts": [int(s) for s in doc_starts]}))
    return out_dir


def feature_fires(cache_dir, index: int, k: int = 8,
                 window: int = 6) -> list[dict]:
    """Top cached contexts for one feature. Reads the cache only.

    Same entry shape as reader.fires_on: activation, token, context,
    mark (char span of the token in the context).
    """
    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    window = int(window)
    if window < 0:
        raise ValueError(f"window must be >= 0, got {window}")
    cache_dir = Path(cache_dir)
    with np.load(cache_dir / FIRES_NAME) as data:
        act = data["act"]
        pos = data["pos"]
    feats, cached_k = act.shape
    index = int(index)
    if not 0 <= index < feats:
        raise ValueError(
            f"feature {index} out of the cached bounds [0,{feats})")
    sample = json.loads((cache_dir / TOKENS_NAME).read_text())
    tokens = sample["tokens"]
    starts = [int(s) for s in sample["doc_starts"]]
    out: list[dict] = []
    for i in range(min(k, cached_k)):
        at = int(pos[index, i])
        if at < 0 or at >= len(tokens):
            continue
        doc = bisect.bisect_right(starts, at) - 1
        doc_start = starts[doc]
        doc_end = starts[doc + 1] if doc + 1 < len(starts) else len(tokens)
        lo = max(doc_start, at - window)
        hi = min(doc_end, at + window + 1)
        token = tokens[at]
        start = sum(len(t) for t in tokens[lo:at])
        out.append({"activation": float(act[index, i]),
                    "token": token,
                    "context": "".join(tokens[lo:hi]),
                    "mark": [start, start + len(token)]})
    return out


def is_whole_word(token: str) -> bool:
    """One token counts iff letters only after strip, length >= 3."""
    stem = str(token).strip().lower()
    return len(stem) >= 3 and stem.isalpha()


def top_words(pushes: dict, n: int = 3) -> list[str]:
    """Top-n whole-word up tokens (fallback: top-1 up token as is)."""
    ups = [e["token"] for e in (pushes or {}).get("up", [])]
    words = [t for t in ups if is_whole_word(t)][: max(int(n), 1)]
    if words:
        return words
    if ups:
        return [ups[0]]
    raise ValueError("top_words needs at least one up token")


def feature_card(eng, sae: TopkSAE, layer: int, index: int,
                 cache_dir=None, k: int = 12) -> dict:
    """One feature card: pushes, coherence, cached fires (no model for fires).

    live is True when the feature fires at least once in the cache.
    """
    if eng is None or eng.model is None or eng.model_id is None:
        raise RuntimeError("no model loaded")
    vec = feature_direction(sae, index)
    pushes = reader.pushes_of_direction(eng, vec, k=k, layer=int(layer))
    fires, built, count = None, False, 0
    live = True
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        if (cache_dir / FIRES_NAME).exists() and (
                cache_dir / TOKENS_NAME).exists():
            sample = json.loads((cache_dir / TOKENS_NAME).read_text())
            count = len(sample.get("tokens", []))
            fires = feature_fires(cache_dir, index, k=k)
            built = True
            live = bool(fires)
    return {"model_id": eng.model_id, "layer": int(layer),
            "index": int(index), "live": live,
            "pushes_toward": pushes, "fires_on": fires,
            "fires_on_cache": {"built": built, "tokens": count}}


def feature_edit(layer: int, sae: TopkSAE, index: int,
                 scale: float, dose: float) -> engine_mod.VectorEdit:
    """VectorEdit that steers one feature (decoder direction, median scale)."""
    vec = feature_direction(sae, index)
    return engine_mod.VectorEdit(
        layer=int(layer), direction=tuple(vec.tolist()),
        scale=float(scale), dose=float(dose))
