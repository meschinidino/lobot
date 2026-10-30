"""Trait families + store (D12). Create accepts validated families only.

A family bundles the validated template (pos/neg instruction pair) and
the hand-validated judge rubric for one trait. Any description that does
not resolve to a family is answered "not supported yet" instead of
running an undefined pipeline.

Families seed from the spike BEHAVIORS (scripts/spike_knob.py) as
unvalidated; mark_validated() records the per-family O1 judge-agreement
gate. Create requires hand_validated=True.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_PATH = ROOT / "store" / "traits.json"

NOT_SUPPORTED = "not supported yet"


class UnsupportedTrait(ValueError):
    """Raised when a description matches no validated family."""


@dataclass(frozen=True)
class TraitFamily:
    name: str
    aliases: tuple = ()
    pos: str = ""
    neg: str = ""
    judge: str = ""
    hand_validated: bool = False
    validation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["aliases"] = list(self.aliases)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TraitFamily":
        return cls(
            name=d["name"],
            aliases=tuple(d.get("aliases", ())),
            pos=d.get("pos", ""),
            neg=d.get("neg", ""),
            judge=d.get("judge", ""),
            hand_validated=bool(d.get("hand_validated", False)),
            validation=dict(d.get("validation", {})),
        )


def _load_behaviors() -> dict:
    spec = importlib.util.spec_from_file_location(
        "spike_knob_mod", ROOT / "scripts" / "spike_knob.py")
    spike = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spike)
    return spike.BEHAVIORS


def seed_families() -> list[TraitFamily]:
    """One unvalidated family per spike behavior (template + rubric)."""
    fams = []
    for name, b in _load_behaviors().items():
        fams.append(TraitFamily(
            name=name,
            aliases=(),
            pos=b["pos"],
            neg=b["neg"],
            judge=b["judge"],
            hand_validated=False,
            validation={"hand_labels": len(b.get("probe", [])),
                        "agreement": None,
                        "note": "O1 gate not run for this family yet"},
        ))
    return fams


class TraitStore:
    """JSON-backed family registry. Create resolves through here."""

    def __init__(self, path: Path | None = None,
                 seed: bool = True):
        self.path = Path(path) if path else DEFAULT_PATH
        self._fams: dict[str, TraitFamily] = {}
        if self.path.exists():
            self.load()
        elif seed:
            for fam in seed_families():
                self._fams[fam.name] = fam

    def all(self) -> list[TraitFamily]:
        return [self._fams[k] for k in sorted(self._fams)]

    def validated(self) -> list[TraitFamily]:
        return [f for f in self.all() if f.hand_validated]

    def get(self, name: str) -> TraitFamily | None:
        return self._fams.get(name)

    def resolve(self, description: str) -> TraitFamily:
        """Match a description to a family (name or alias, case-free).

        Raises UnsupportedTrait("... not supported yet") for anything
        else, including validated-only misses: Create must not run an
        undefined pipeline.
        """
        norm = description.strip().lower()
        for fam in self._fams.values():
            if norm == fam.name.lower() or norm in [
                    a.lower() for a in fam.aliases]:
                return fam
        supported = ", ".join(f.name for f in self.validated()) or "none"
        raise UnsupportedTrait(
            f"'{description}' is {NOT_SUPPORTED}. "
            f"Supported families: {supported}.")

    def register(self, fam: TraitFamily, overwrite: bool = False) -> None:
        if fam.name in self._fams and not overwrite:
            raise ValueError(f"family '{fam.name}' already registered")
        self._fams[fam.name] = fam

    def mark_validated(self, name: str, *, n_labels: int,
                       agreement: float, note: str = "") -> TraitFamily:
        """Record the per-family O1 judge-agreement gate (hand labels)."""
        fam = self._fams.get(name)
        if fam is None:
            raise KeyError(f"no family '{name}'")
        updated = TraitFamily(
            name=fam.name, aliases=fam.aliases, pos=fam.pos,
            neg=fam.neg, judge=fam.judge, hand_validated=True,
            validation={"hand_labels": n_labels, "agreement": agreement,
                        "note": note})
        self._fams[name] = updated
        return updated

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"families": [f.to_dict() for f in self.all()]}
        self.path.write_text(json.dumps(payload, indent=2) + "\n")

    def load(self) -> None:
        payload = json.loads(self.path.read_text())
        self._fams = {d["name"]: TraitFamily.from_dict(d)
                      for d in payload.get("families", [])}
