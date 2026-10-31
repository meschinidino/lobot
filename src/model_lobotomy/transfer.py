"""Cross-model transfer (1000X XM): carry a trait to another model.

A trait family (pos, neg, judge, baselines) does not depend on the
model, so a transfer re-derives the direction on the target model
with the same seed, splits and judge (one shared benchmark, O7) and
records evidence on both sides. A target that fails its gates is
"unavailable" with the gate that failed, never hidden.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from model_lobotomy import patch as patch_mod

RECORD_VERSION = "lobot-transfer-v1"


def unavailable_reason(report: dict) -> str | None:
    """None when the trait is a knob on this model, else the reason."""
    verdict = report.get("verdict")
    if verdict == "knob":
        return None
    if verdict == "blocked":
        return "the guard blocks the derived patch"
    if report.get("frozen") is None:
        return ("tuning found no coherent dose (no layer and dose "
                "raised the judge without breaking fluency)")
    elig = report.get("eligibility") or {}
    failed = [v for v in ("read", "update_up", "update_down", "delete")
              if not elig.get(v)]
    return "no verb passed its gate: " + ", ".join(failed)


def side_summary(report: dict) -> dict:
    """The evidence one side contributes to a transfer record."""
    fp = report.get("fingerprint") or {}
    up = ((report.get("final") or {}).get("update_up") or {}).get("card")
    frozen = report.get("frozen") or {}
    return {"model_id": fp.get("model_id"),
            "revision": fp.get("revision"),
            "verdict": report.get("verdict"),
            "layer": frozen.get("layer"), "dose": frozen.get("dose"),
            "tuning_gain": frozen.get("tuning_gain"),
            "eligibility": report.get("eligibility"),
            "read_agreement": ((report.get("read_gate") or {})
                               .get("agreement") or {}).get("fraction"),
            "update_up_card": up,
            "seconds": report.get("seconds"),
            "unavailable": unavailable_reason(report)}


def transfer_record(trait: str, judge: str, source_report: dict,
                    target_report: dict,
                    source_patch=None, target_patch=None) -> dict:
    """Both sides' evidence on one shared benchmark (same seed, judge).

    Raises ValueError when the two runs did not share the benchmark:
    different trait, seed or final split prompts.
    """
    for key in ("trait", "seed"):
        if source_report.get(key) != target_report.get(key):
            raise ValueError(f"runs differ in {key}: "
                             f"{source_report.get(key)!r} vs "
                             f"{target_report.get(key)!r}")
    fin_a = (source_report.get("splits") or {}).get("final")
    fin_b = (target_report.get("splits") or {}).get("final")
    if fin_a != fin_b:
        raise ValueError("runs used different final prompts")
    rec = {"version": RECORD_VERSION, "trait": trait, "judge": judge,
           "seed": source_report.get("seed"),
           "benchmark_sha256": patch_mod.sha256_hex(
               patch_mod.canonical_json({"final": fin_a,
                                         "judge": judge})),
           "source": side_summary(source_report),
           "target": side_summary(target_report),
           "created": round(time.time(), 3)}
    for side, path in (("source", source_patch), ("target", target_patch)):
        if path is not None and Path(path).is_file():
            rec[side]["patch_sha256"] = patch_mod.sha256_hex(
                Path(path).read_bytes())
    rec["status"] = ("transferred" if rec["target"]["unavailable"] is None
                     else "unavailable")
    return rec


def save_record(path, rec: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2))
    return path
