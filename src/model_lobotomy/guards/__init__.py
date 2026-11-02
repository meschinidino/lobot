"""Guard port (MOD phase 5): screen (fast) and confirm (full).

Setting: LOBOT_GUARD (default `screens-v2`; `noop` for tests).
The public verdict shape is GuardPortVerdict. Old reports keep
their shape: `verdict_dict` in guard.py is unchanged, and
`from_old` maps an old verdict into the port shape. Guard-specific
numbers live in `measure`; `measure_text` is the plain sentence.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GuardPortVerdict:
    blocked: bool = False
    would_block: bool = False
    verified: bool = False
    enforced: bool = False
    research_mode: bool = False
    screens: dict = field(default_factory=dict)
    raw: object = None

    def to_dict(self) -> dict:
        return {"blocked": self.blocked, "would_block": self.would_block,
                "verified": self.verified, "enforced": self.enforced,
                "research_mode": self.research_mode,
                "screens": self.screens}


def _screen_port(name, d: dict) -> dict:
    measure = {k: d[k] for k in ("min_ratio", "limit", "floor", "worst_prompt",
                                  "rule", "flips", "control_flips") if k in d}
    return {"blocked": d.get("blocked", False),
            "would_block": d.get("would_block", False),
            "flagged_prompts": list(d.get("blocked_prompts") or []),
            "measure_text": d.get("display_prompt") or "",
            "measure": measure}


def from_old(gv, enforced: bool = False) -> GuardPortVerdict:
    """Map guard.GuardVerdict (or its dict) into the port shape."""
    from model_lobotomy import guard as _g
    d = gv if isinstance(gv, dict) else _g.verdict_dict(gv)
    screens = {name: _screen_port(name, sd)
               for name, sd in (d.get("screens") or {}).items()}
    if not screens and "min_ratio" in d:
        screens = {"refusal": _screen_port("refusal", d)}
    return GuardPortVerdict(
        blocked=bool(d.get("blocked")), would_block=bool(d.get("would_block")),
        verified=bool(d.get("verified")), enforced=enforced,
        research_mode=bool(getattr(gv, "research_mode", False)),
        screens=screens, raw=gv)


class ScreensV2Guard:
    """Today's guard, wrapped. screen=fast, confirm=full generation."""
    name = "screens-v2"
    def __init__(self, inner=None):
        from model_lobotomy import guard as _g
        self._inner = inner or _g.Guard()
    # Old surface, forwarded (callers migrate at their own pace).
    def check(self, *a, **kw): return self._inner.check(*a, **kw)
    def check_verified(self, *a, **kw): return self._inner.check_verified(*a, **kw)
    def check_confirmed(self, *a, **kw): return self._inner.check_confirmed(*a, **kw)
    def check_full(self, *a, **kw): return self._inner.check_full(*a, **kw)
    def check_screen(self, *a, **kw): return self._inner.check_screen(*a, **kw)
    # Port surface.
    def screen(self, engine, edits, screens=None, **kw) -> GuardPortVerdict:
        gv = self._inner.check(engine, edits, **kw)
        return from_old(gv)
    def confirm(self, engine, edits, screens=None, **kw) -> GuardPortVerdict:
        gv = self._inner.check_confirmed(engine, edits, **kw)
        return from_old(gv)


class NoopGuard:
    """Test-only guard: never blocks. Proves a swap needs no caller change."""
    name = "noop"
    def screen(self, engine, edits, screens=None, **kw):
        return GuardPortVerdict()
    def confirm(self, engine, edits, screens=None, **kw):
        return GuardPortVerdict(verified=True)


def get_guard(name=None):
    from model_lobotomy.config import get_settings
    kind = name or get_settings().guard_kind
    if kind == "noop":
        return NoopGuard()
    if kind == "screens-v2":
        return ScreensV2Guard()
    raise ValueError(f"unknown LOBOT_GUARD {kind!r}; want screens-v2 or noop")


REGISTRY = ("screens-v2", "noop")
