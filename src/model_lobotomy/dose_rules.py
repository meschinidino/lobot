"""Dose-rule port (MOD phase 6): which dose a tuning table picks.

Setting: LOBOT_DOSE_RULE (default `fluency-first-v3`).
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


class FluencyFirstRule:
    """Highest judge score among doses that stay fluent per prompt.

    The NLL cap in nll+fluency-v1 held weak traits under their
    effective dose (cowboy tuning: dose 1.0 scores 0.81 with fluency
    0.75, but its NLL rise of 1.47 broke the cap of 1.0, so the knob
    stayed at 0.5). Fluency is the direct coherence signal: repeated
    tokens score near zero. NLL stays in the table as a cost report,
    not a gate.

    v3 adds a per-prompt gate: the worst-decile fluency (fluency_p10
    in the table) must also pass. Mean fluency alone passed
    shakespeare at dose 1.0, where some prompts loop words
    (fluency 0.005) while others stay clean. Tables without
    fluency_p10 (older reports) skip the per-prompt gate.
    """
    name = "fluency-first-v3"

    def __init__(self, fluency_floor: float = 0.5,
                 fluency_p10_floor: float = 0.3, name: str | None = None):
        self.fluency_floor = fluency_floor
        self.fluency_p10_floor = fluency_p10_floor
        if name is not None:
            self.name = name

    @property
    def params(self) -> dict:
        return {"fluency_floor": self.fluency_floor,
                "fluency_p10_floor": self.fluency_p10_floor}

    def coherent(self, gen: dict) -> list[float]:
        return [d for d in gen if d > 0
                and gen[d]["fluency"] >= self.fluency_floor
                and gen[d].get("fluency_p10", 1.0)
                >= self.fluency_p10_floor]

    def choose_gen(self, gen: dict):
        return max(self.coherent(gen), key=lambda d: gen[d]["judge"],
                   default=None)

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
    if kind == "fluency-first-v3":
        return FluencyFirstRule()
    if kind == "fluency-first-v2":
        return FluencyFirstRule(fluency_p10_floor=0.0,
                                name="fluency-first-v2")
    if kind == "nll+fluency-v1":
        return NllFluencyRule()
    if kind == "fake":
        return FakeDoseRule()
    raise ValueError(f"unknown LOBOT_DOSE_RULE {kind!r}")


REGISTRY = ("fluency-first-v3", "fluency-first-v2",
            "nll+fluency-v1", "fake")
