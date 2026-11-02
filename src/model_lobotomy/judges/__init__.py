"""Judge port (MOD phase 4): one score_many for every judge.

Setting: LOBOT_JUDGE (default `local`; `meta_api`, `fake`).
Every implementation passes the same contract tests. The memo
cache lives here: score_many dedupes pairs per instance.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class JudgePort(Protocol):
    name: str
    def score_many(self, pairs, label) -> list[float]: ...


class MemoJudge:
    """Base: memoizes score_many over a raw batch scorer."""
    name = "?"
    def __init__(self, raw):
        self._raw = raw  # raw(keys: list[(q, r, label)]) -> list[float]
        self._cache: dict[tuple, float] = {}

    def score_many(self, pairs, label) -> list[float]:
        keys = [(q, r, label) for q, r in pairs]
        todo = list(dict.fromkeys(k for k in keys if k not in self._cache))
        if todo:
            self._cache.update(zip(todo, self._raw(todo)))
        return [self._cache[k] for k in keys]

    def score(self, question: str, reply: str, rubric: str) -> float:
        return self.score_many([(question, reply)], rubric)[0]


class LocalJudgePort(MemoJudge):
    name = "local"
    def __init__(self, score_one):
        super().__init__(lambda keys: [score_one(q, r, lab) for q, r, lab in keys])


class FakeJudge(MemoJudge):
    """Test double: 1.0 when the label's words appear in the reply."""
    name = "fake"
    def __init__(self):
        super().__init__(self._score)
    @staticmethod
    def _score(keys):
        out = []
        for _q, reply, label in keys:
            words = [w for w in (label or "").lower().split() if len(w) > 3]
            hit = any(w in (reply or "").lower() for w in words)
            out.append(1.0 if hit else 0.0)
        return out


def get_judge(name=None, *, score_one=None, raw=None):
    """Factory. `local` needs score_one or raw; `meta_api` wraps trait_eval."""
    from model_lobotomy.config import get_settings
    kind = name or get_settings().judge_kind
    if kind == "fake":
        return FakeJudge()
    if kind == "meta_api":
        from model_lobotomy import trait_eval
        api = trait_eval.MetaApiJudge()
        return LocalJudgePort(api.score) if False else MemoJudge(
            lambda keys: [api.score(q, r, lab) for q, r, lab in keys])
    if kind == "local":
        if raw is not None:
            j = MemoJudge(raw); j.name = "local"; return j
        if score_one is None:
            raise ValueError("local judge needs a bound score_one")
        return LocalJudgePort(score_one)
    raise ValueError(f"unknown LOBOT_JUDGE {kind!r}; want local, meta_api or fake")


REGISTRY = ("local", "meta_api", "fake")
