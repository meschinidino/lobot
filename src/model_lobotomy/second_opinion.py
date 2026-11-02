"""Second opinion: Muse Spark re-scores the turn-up pairs (J1).

The local judge still gates Create. This module only re-scores the
recorded base and steered replies with the Meta API judge. It
changes no gate and no verdict.

At most MAX_WORKERS replies score at one time. Each worker thread
holds one judge, so the call and token counts stay exact.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor

from model_lobotomy import evaluation

MAX_WORKERS = 8


def pairs_hash(pairs: list) -> str:
    """Hash of the pairs. The cache key for one second opinion."""
    body = json.dumps(pairs, sort_keys=True).encode()
    return hashlib.sha256(body).hexdigest()


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def run_second_opinion(pairs: list, label: str, judge_factory) -> dict:
    """Re-score pairs with a fresh judge per worker thread.

    judge_factory() returns one judge with score(q, reply, label),
    a usage dict and a model name. Pairs hold q, base, steered and
    the local scores jb and js.
    """
    local = threading.local()
    judges, judges_lock = [], threading.Lock()

    def judge():
        if not hasattr(local, "judge"):
            local.judge = judge_factory()
            with judges_lock:
                judges.append(local.judge)
        return local.judge

    def task(item):
        i, side, q, reply = item
        return i, side, judge().score(q, reply, label)

    items = []
    for i, p in enumerate(pairs):
        items.append((i, "base", p["q"], p["base"]))
        items.append((i, "steered", p["q"], p["steered"]))
    scored: dict = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for i, side, value in pool.map(task, items):
            scored[(i, side)] = float(value)

    muse_gains, local_gains = [], []
    agree = 0
    for i, p in enumerate(pairs):
        muse_gain = scored[(i, "steered")] - scored[(i, "base")]
        local_gain = float(p["js"]) - float(p["jb"])
        muse_gains.append(muse_gain)
        local_gains.append(local_gain)
        agree += _sign(muse_gain) == _sign(local_gain)

    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    for j in judges:
        for k in usage:
            usage[k] += int((getattr(j, "usage", None) or {}).get(k, 0))
    lo, hi = evaluation.bootstrap_ci(muse_gains)
    return {
        "muse_mean_gain": round(statistics.fmean(muse_gains), 3),
        "muse_ci95": [round(lo, 3), round(hi, 3)],
        "local_mean_gain": round(statistics.fmean(local_gains), 3),
        "agree": agree,
        "total": len(pairs),
        "calls": usage["calls"],
        "tokens": usage["input_tokens"] + usage["output_tokens"],
        "model": getattr(judges[0], "model", None) if judges else None,
    }
