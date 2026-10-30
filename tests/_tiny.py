"""A tiny real torch model shaped like Llama, for hook and capture tests (no weights)."""
from __future__ import annotations

import torch


class TinyBlock(torch.nn.Module):
    """Residual block: x + down(relu(up(x))); up is hidden->intermediate, down back."""

    def __init__(self, hidden, inter):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.up_proj = torch.nn.Linear(hidden, inter, bias=False)
        self.mlp.down_proj = torch.nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return x + self.mlp.down_proj(torch.relu(self.mlp.up_proj(x)))


class TinyModel(torch.nn.Module):
    """model.model.layers[i].mlp.down_proj: the "llama" block path in steer."""

    def __init__(self, layers=2, hidden=4, inter=6, vocab=10):
        super().__init__()
        torch.manual_seed(0)
        self.config = type("Cfg", (), {"hidden_size": hidden, "intermediate_size": inter,
                                       "num_hidden_layers": layers})()
        self.model = torch.nn.Module()
        self.model.embed = torch.nn.Embedding(vocab, hidden)
        self.model.layers = torch.nn.ModuleList(TinyBlock(hidden, inter) for _ in range(layers))

    def forward(self, ids):
        x = self.model.embed(ids)
        for blk in self.model.layers:
            x = blk(x)
        return x


class TinyTok:
    """Chat template = the text; one token per word."""

    def apply_chat_template(self, msgs, **_):
        return msgs[0]["content"]

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[len(w) % 10 for w in text.split()]])}
