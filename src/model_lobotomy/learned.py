"""Learned steering vectors (ML v3; specs/ML-multi-layer-search.md).

One trainable vector per chosen layer, added to the block output at
every position (the same place engine hooks add). The model stays
frozen; Adam moves only the vectors. Loss per contrast pair:

    NLL(positive answer) - NEG_W * clamp(NLL(negative answer))
    + DRIFT_W * KL(plain || steered) on the question tokens

The drift term is the fluency guard: without it, training finds the
same broken text the dose sweep found. Tokenizer-free: callers pass
tokenized sequences {"input_ids", "answer_start"} (loss reads the
answer span) and drift sequences {"input_ids"} (prompt-only).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

NEG_W = 0.1
NEG_CAP = 6.0
DRIFT_W = 1.0
LR = 0.05
BATCH_PAIRS = 5
EPOCHS = 15


def fewshot_prompt(instruction: str, shots: list, question: str) -> str:
    """Instruction plus worked examples, then the real question.

    The plain model answers a bare "be rude" instruction with a
    smug, watered-down reply: its alignment caps the style. Worked
    examples show the target style instead of naming it, and the
    model's answers reach the style the instruction only names.
    Training then learns from answers at the target strength.
    """
    if not shots:
        return f"{instruction}\n\n{question}"
    lines = [instruction, ""]
    for s in shots:
        lines += [f"Question: {s['q']}", f"Answer: {s['a']}", ""]
    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


def _add_hook(v):
    def hook(_mod, _args, out):
        if isinstance(out, torch.Tensor):
            h = out
            rest = None
        else:
            h, rest = out[0], tuple(out[1:])
        h = h + v.to(dtype=h.dtype)
        return h if rest is None else (h,) + rest
    return hook


def _nll(logits, ids, start):
    """Mean NLL of tokens from `start` (predicted by prior tokens)."""
    tgt = ids[1:]
    pred = logits[:-1]
    mask = torch.arange(len(tgt), device=ids.device) >= (start - 1)
    if not mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(pred[mask].float(), tgt[mask])


def _forward(model, ids):
    out = model(input_ids=ids.unsqueeze(0))
    return out.logits[0] if hasattr(out, "logits") else out[0]


def train_vectors(model, layer_modules, sequences, drift_sequences,
                  init, *, epochs=EPOCHS, batch_pairs=BATCH_PAIRS,
                  lr=LR, neg_w=NEG_W, drift_w=DRIFT_W, seed=0,
                  log=None):
    """Train vectors. Returns ({layer: cpu tensor}, history).

    layer_modules: {layer: module}. init: {layer: tensor} start
    vectors (same shape as the block output width). Model params
    are frozen by the caller and never touched here.
    """
    if not drift_sequences:
        raise ValueError("drift_sequences must not be empty")
    for s in sequences:
        if s["answer_start"] >= len(s["input_ids"]):
            raise ValueError("sequence has an empty answer span")
    # Batch by contrast pair, never by loose sequence: a batch with
    # only one side has no push-down (or no push-up) signal. Bad
    # pairs are an error, never a silent drop.
    by_pair: dict = {}
    for s in sequences:
        slot = by_pair.setdefault(s["pair"], {})
        if s["side"] in slot:
            raise ValueError(f"pair {s['pair']} has two {s['side']} "
                             f"sequences")
        slot[s["side"]] = s
    pairs = list(by_pair.values())
    if not pairs or any(set(v) != {"pos", "neg"} for v in pairs):
        raise ValueError("every pair needs one pos and one neg "
                         "sequence")
    device = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    vecs = {L: torch.nn.Parameter(
        torch.as_tensor(v, dtype=torch.float32, device=device).clone())
        for L, v in init.items()}
    opt = torch.optim.Adam(list(vecs.values()), lr=lr)
    hist = []
    handles: list = []
    try:
        # Register one by one into the live list: a failure halfway
        # still leaves every live hook tracked for the finally.
        for L in vecs:
            handles.append(layer_modules[L].register_forward_hook(
                _add_hook(vecs[L])))
        for ep in range(epochs):
            order = torch.randperm(len(pairs), generator=g).tolist()
            for i in range(0, len(order), batch_pairs):
                batch = [s for j in order[i:i + batch_pairs]
                         for s in pairs[j].values()]
                opt.zero_grad()
                pos_l = neg_l = None
                for seq in batch:
                    ids = seq["input_ids"].to(device)
                    logits = _forward(model, ids)
                    n = _nll(logits, ids, seq["answer_start"])
                    if seq["side"] == "pos":
                        pos_l = n if pos_l is None else pos_l + n
                    else:
                        neg_l = n if neg_l is None else neg_l + n
                np_, nn_ = max(1, sum(s["side"] == "pos" for s in batch)), \
                    max(1, sum(s["side"] == "neg" for s in batch))
                loss = pos_l / np_ - neg_w * torch.clamp(neg_l / nn_,
                                                         max=NEG_CAP)
                # Drift: KL(plain || steered) on prompt-only inputs.
                kl_total = None
                for seq in drift_sequences:
                    ids = seq["input_ids"].to(device)
                    with torch.no_grad():
                        for h in handles:
                            h.remove()
                        handles.clear()
                        plain = _forward(model, ids)
                        for L in vecs:
                            handles.append(
                                layer_modules[L].register_forward_hook(
                                    _add_hook(vecs[L])))
                    steered = _forward(model, ids)
                    kl = F.kl_div(
                        F.log_softmax(steered[:-1].float(), -1),
                        F.softmax(plain[:-1].float(), -1),
                        reduction="batchmean")
                    kl_total = kl if kl_total is None else kl_total + kl
                loss = loss + drift_w * kl_total / max(
                    1, len(drift_sequences))
                loss.backward()
                opt.step()
                row = {"epoch": ep,
                       "pos_nll": round(float((pos_l / np_).detach()), 4),
                       "neg_nll": round(float((neg_l / nn_).detach()), 4),
                       "drift_kl": round(float(kl_total.detach()) /
                                         max(1, len(drift_sequences)), 4),
                       "loss": round(float(loss.detach()), 4)}
                hist.append(row)
                if log:
                    log(row)
    finally:
        for h in handles:
            h.remove()
    return {L: v.detach().cpu() for L, v in vecs.items()}, hist
