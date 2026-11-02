"""Dose-rule port (MOD phase 6): which dose a tuning table picks.

Setting: LOBOT_DOSE_RULE (default `nll+fluency-v1`).
A rule has `name`, `params` (written into the report) and
`choose_gen(gen) -> dose or None` for one layer's dose table.
`choose(table)` picks (layer, dose) by judge gain, as tune did.
"""
from __future__ import annotations


class NllFluencyRule:
    name = "nll+fluency-v1"
    def __init__(self, nll_max_rise: float = 1.0, fluency_floor: float = 0.5):
        self.nll_max_rise = nll_max_rise
        self.fluency_floor = fluency_floor
    @property
    def params(self) -> dict:
        return {"nll_max_rise": self.nll_max_rise,
                "fluency_floor": self.fluency_floor}
    def coherent(self, gen: dict) -> list[float]:
        base = gen[0.0]["nll"]
        return [d for d in gen if d > 0
                and gen[d]["nll"] - base <= self.nll_max_rise
                and gen[d]["fluency"] >= self.fluency_floor]
    def choose_gen(self, gen: dict):
        return max(self.coherent(gen), key=lambda d: gen[d]["judge"], default=None)
    def choose(self, table: dict):
        best = None
        for layer, gen in table.items():
            dose = self.choose_gen(gen)
            if dose is None:
                continue
            gain = gen[dose]["judge"] - gen[0.0]["judge"]
            if best is None or gain > best[2]:
                best = (layer, dose, gain)
        return None if best is None else (best[0], best[1])


class FakeDoseRule(NllFluencyRule):
    """Test double: every positive dose is coherent."""
    name = "fake"
    @property
    def params(self): return {}
    def coherent(self, gen): return [d for d in gen if d > 0]


def get_dose_rule(name=None):
    from model_lobotomy.config import get_settings
    kind = name or get_settings().dose_rule_kind
    if kind == "nll+fluency-v1":
        return NllFluencyRule()
    if kind == "fake":
        return FakeDoseRule()
    raise ValueError(f"unknown LOBOT_DOSE_RULE {kind!r}")


REGISTRY = ("nll+fluency-v1", "fake")
