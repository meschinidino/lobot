"""lobot evaluation gate: pure helpers, no model calls.

Split discipline for a prompt bank, percentile bootstrap confidence
intervals, the four-rule report card, and a ledger that enforces one
final evaluation per frozen candidate version.

Covered by tests/test_evaluation.py (fixed fixtures, boundary-proven).
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Splits:
    """Disjoint deterministic split of a prompt bank."""
    extraction: list
    tuning: list
    final: list
    reserve: list


def make_splits(prompts, seed: int, *, n_extract: int = 60,
                n_tune: int = 30, n_final: int = 30) -> Splits:
    """Split a >=120-prompt bank into 60 extraction / 30 tuning / 30 final.

    Deterministic per seed; the three splits are disjoint; leftover
    prompts form the reserve pool for retries. Raises ValueError when
    the bank is too small to fill the three splits.
    """
    prompts = list(prompts)
    need = n_extract + n_tune + n_final
    if len(prompts) < need:
        raise ValueError(
            f"prompt bank too small: {len(prompts)} < {need} "
            f"({n_extract}+{n_tune}+{n_final})")
    rng = random.Random(seed)
    order = list(range(len(prompts)))
    rng.shuffle(order)
    by_role = [prompts[i] for i in order]
    e = by_role[:n_extract]
    t = by_role[n_extract:n_extract + n_tune]
    f = by_role[n_extract + n_tune:need]
    return Splits(extraction=e, tuning=t, final=f, reserve=by_role[need:])


def bootstrap_ci(values, n: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. Returns (lo, hi).

    Resamples len(values) with replacement n times (seeded), takes the
    mean of each resample, and returns the alpha/2 and 1-alpha/2
    percentiles. Raises ValueError on empty input.
    """
    values = list(values)
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    if n < 1:
        raise ValueError(f"bootstrap_ci needs n >= 1, got {n}")
    rng = random.Random(seed)
    m = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=m)) for _ in range(n))
    lo_idx = min(n - 1, int((alpha / 2) * n))
    hi_idx = max(0, int((1 - alpha / 2) * n) - 1)
    return (means[lo_idx], means[hi_idx])


def _excludes_zero(ci: tuple[float, float]) -> bool:
    lo, hi = ci
    return hi < 0 or lo > 0


def report_card_pass(trait_gains, control_gains, *, n: int = 10000,
                     seed: int = 0, min_gain: float = 0.30,
                     control_max: float = 0.10) -> dict:
    """Four-rule gate. Returns {"passed", "rules", "trait_ci", ...}.

    Pass requires ALL of:
      1. trait gain CI excludes 0,
      2. mean trait gain >= 0.30,
      3. control upper 95% bound < 0.10,
      4. paired (trait - control) CI excludes 0.
    trait_gains and control_gains must be non-empty and equal-length
    (paired); ValueError otherwise.
    """
    trait_gains = list(trait_gains)
    control_gains = list(control_gains)
    if not trait_gains or not control_gains:
        raise ValueError("report_card_pass needs non-empty gain lists")
    if len(trait_gains) != len(control_gains):
        raise ValueError(
            f"paired gains must match in length: "
            f"{len(trait_gains)} != {len(control_gains)}")
    trait_ci = bootstrap_ci(trait_gains, n=n, seed=seed)
    control_ci = bootstrap_ci(control_gains, n=n, seed=seed + 1)
    paired = [t - c for t, c in zip(trait_gains, control_gains)]
    paired_ci = bootstrap_ci(paired, n=n, seed=seed + 2)
    trait_mean = statistics.fmean(trait_gains)
    rules = {
        "trait_ci_excludes_zero": _excludes_zero(trait_ci),
        "trait_mean_ge_min": trait_mean >= min_gain,
        "control_upper_lt_max": control_ci[1] < control_max,
        "paired_ci_excludes_zero": _excludes_zero(paired_ci),
    }
    return {
        "passed": all(rules.values()),
        "rules": rules,
        "trait_ci": trait_ci,
        "trait_mean": trait_mean,
        "control_ci": control_ci,
        "paired_ci": paired_ci,
    }


@dataclass
class FinalSplitLedger:
    """One final evaluation per frozen candidate version.

    A frozen version is (candidate_id, instructions, layer, dose).
    record_final raises ValueError on a second final for the same
    frozen version. reserve_for_retry hands out fresh reserve prompts
    per candidate (never repeats one already handed out) and raises
    ValueError when the pool is exhausted.
    """
    _finals: dict = field(default_factory=dict)
    _used_reserve: dict = field(default_factory=dict)

    def record_final(self, candidate_id: str, *, instructions: str,
                     layer: int, dose: float, prompts: list) -> dict:
        """Record a final; reject a second final for the same version."""
        key = (candidate_id, instructions, layer, dose)
        if key in self._finals:
            raise ValueError(
                f"second final rejected for frozen {key!r}: finals run once; "
                f"freeze a new version or retry from reserve prompts")
        record = {"candidate_id": candidate_id, "instructions": instructions,
                  "layer": layer, "dose": dose, "prompts": list(prompts)}
        self._finals[key] = record
        return record

    def reserve_for_retry(self, candidate_id: str, reserve_pool: list,
                          n: int) -> list:
        """Hand out n fresh reserve prompts for a retry (never repeats)."""
        if n < 1:
            raise ValueError(f"reserve_for_retry needs n >= 1, got {n}")
        used = self._used_reserve.setdefault(candidate_id, set())
        fresh = [p for p in reserve_pool if p not in used]
        if len(fresh) < n:
            raise ValueError(
                f"reserve exhausted for {candidate_id!r}: "
                f"need {n}, only {len(fresh)} fresh left")
        out = fresh[:n]
        used.update(out)
        return out
