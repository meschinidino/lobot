"""Guard port (MOD phase 5): screen (fast) and confirm (full).

Setting: LOBOT_GUARD (default `screens-v2`; `noop` for tests).
The public verdict shape is GuardPortVerdict. Callers use only
`get_guard()` plus `screen()` or `confirm()` and read that verdict.
`report` and `screen_reports` hold the old JSON dict shapes.
Reports, streams and patch files keep those shapes.
Guard-specific numbers live in each screen's `measure`.
`measure_text` is the plain sentence for that screen.
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
    cached: bool = False
    report: dict = field(default_factory=dict)
    screen_reports: dict = field(default_factory=dict)

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


def _clean_report(verified: bool) -> dict:
    return {"blocked": False, "would_block": False,
            "verified": verified, "min_ratio": 1.0,
            "worst_prompt": None, "blocked_prompts": [],
            "limit": 0.5, "floor": 0.2, "display_prompt": ""}


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
        screens=screens, cached=bool(getattr(gv, "cached", False)),
        report=d, screen_reports=dict(d.get("screens") or {}))


class ScreensV2Guard:
    """Today's guard behind the port. screen is the fast prefill check.

    confirm runs the full path. With `screens=[...]` it runs full
    generation on those screens only. With `confirmed=True` it runs
    the always-full confirmed check (never blocks the call).
    Without either, it runs the verified check: a prefill block
    returns at once, a prefill allow is confirmed by generation.
    """
    name = "screens-v2"

    def __init__(self, inner=None):
        from model_lobotomy import guard as _g
        self._inner = inner or _g.Guard()

    def _inner_for(self, kw):
        refusal_set = kw.pop("refusal_set", None)
        if refusal_set is None:
            return self._inner
        from model_lobotomy import guard as _g
        return _g.Guard(refusal_set=tuple(refusal_set))

    def screen(self, engine, edits, screens=None, **kw) -> GuardPortVerdict:
        gv = self._inner_for(kw).check(engine, edits, **kw)
        return from_old(gv)

    def confirm(self, engine, edits, screens=None, **kw) -> GuardPortVerdict:
        inner = self._inner_for(kw)
        if screens is not None:
            return self._confirm_screens(engine, edits, list(screens), inner, **kw)
        if kw.pop("confirmed", False):
            gv = inner.check_confirmed(engine, edits, **kw)
            return from_old(gv)
        gv = inner.check_verified(engine, edits, **kw)
        return from_old(gv)

    def _confirm_screens(self, engine, edits, screens, inner, **kw):
        from model_lobotomy import guard as _g
        from model_lobotomy import guard_screens as _screens
        entries, reports = {}, {}
        for name in screens:
            if name == "refusal":
                full = inner.check_full(engine, edits, **kw)
                sv = _g.refusal_entry(full)
            else:
                sv = inner.check_screen(engine, edits, name, **kw)
            report = _screens.screen_dict(name, sv, _g.SCREEN_DISPLAY[name])
            reports[name] = report
            entries[name] = _screen_port(name, report)
        blocked = any(r["blocked"] for r in reports.values())
        would = any(r["blocked"] or r["would_block"]
                    for r in reports.values())
        report = {"blocked": blocked, "would_block": would,
                  "verified": True, "screens": reports}
        return GuardPortVerdict(blocked=blocked, would_block=would,
                                verified=True, screens=entries,
                                report=report, screen_reports=reports)


class NoopGuard:
    """Test-only guard: never blocks. Proves a swap needs no caller change."""
    name = "noop"

    def screen(self, engine, edits, screens=None, **kw):
        return GuardPortVerdict(report=_clean_report(False))

    def confirm(self, engine, edits, screens=None, **kw):
        entries, reports = {}, {}
        for name in (screens or []):
            reports[name] = {"blocked": False, "would_block": False,
                             "min_ratio": 1.0, "worst_prompt": None,
                             "blocked_prompts": [], "display_prompt": ""}
            entries[name] = _screen_port(name, reports[name])
        report = _clean_report(True)
        if reports:
            report = {**report, "screens": reports}
        return GuardPortVerdict(verified=True, screens=entries,
                                report=report, screen_reports=reports)


def get_guard(name=None):
    from model_lobotomy.config import get_settings
    kind = name or get_settings().guard_kind
    if kind == "noop":
        return NoopGuard()
    if kind == "screens-v2":
        return ScreensV2Guard()
    raise ValueError(f"unknown LOBOT_GUARD {kind!r}; want screens-v2 or noop")


REGISTRY = ("screens-v2", "noop")
