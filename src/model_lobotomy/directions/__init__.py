"""Direction port (MOD phase 6): how a direction is derived.

Setting: LOBOT_DIRECTION (default `diffmean`).
derive(engine, spec, cfg, prompts) -> (A, B) captures, and
seps(A, B, layers) -> {layer: {unit, acc, effect}}.
"""
from __future__ import annotations


class DiffMeanDirection:
    name = "diffmean"
    def derive(self, eng, spec, cfg, prompts):
        A = eng.capture([f"{cfg['pos']}\n\n{q}" for q in prompts],
                        list(range(spec.layers)), point="block_out", reduce="last")
        B = eng.capture([f"{cfg['neg']}\n\n{q}" for q in prompts],
                        list(range(spec.layers)), point="block_out", reduce="last")
        return A, B
    def seps(self, A, B, layers):
        from model_lobotomy import spike
        return {L: spike.separability(A[L][:40], B[L][:40], A[L][40:], B[L][40:])
                for L in layers}


def get_direction(name=None):
    from model_lobotomy.config import get_settings
    kind = name or get_settings().direction_kind
    if kind == "diffmean":
        return DiffMeanDirection()
    raise ValueError(f"unknown LOBOT_DIRECTION {kind!r}; want diffmean")


REGISTRY = ("diffmean",)
