"""Evidence and patch files for saved traits (E1).

Reads finished Create jobs under jobs/<id>/: job.json for status, and
trait/report.json, trait/report.md, trait/knob.lobot for the evidence.
Pure file access; no model. server.py exposes it as /api/traits*.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# Job ids look like "hostility-0-20261030-221500-000": no dots, no slashes.
JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
CARD_KEYS = ("passed", "trait_mean", "trait_ci")


def job_dir(jobs_dir: Path, job_id: str) -> Path | None:
    """The job's folder, or None when the id is malformed or escapes jobs_dir."""
    if not JOB_ID.match(job_id):
        return None
    root = Path(jobs_dir).resolve()
    d = (root / job_id).resolve()
    if d.parent != root or not (d / "job.json").is_file():
        return None
    return d


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _card(card: dict | None) -> dict | None:
    return {k: card.get(k) for k in CARD_KEYS} if card else None


def list_traits(jobs_dir: Path, model_id: str | None = None) -> list[dict]:
    """Finished jobs as trait items (contracts section 4), newest first."""
    jobs_dir = Path(jobs_dir)
    done = []
    if jobs_dir.is_dir():
        for d in jobs_dir.iterdir():
            job = _read(d / "job.json")
            if job.get("status") == "done" and JOB_ID.match(d.name):
                done.append((d, job))
    done.sort(key=lambda x: x[1].get("created", 0))
    versions: dict[tuple, int] = {}
    items = []
    for d, job in done:
        key = (job.get("trait"), job.get("model_id"))
        versions[key] = versions.get(key, 0) + 1
        if model_id and job.get("model_id") != model_id:
            continue
        rep = _read(d / "trait" / "report.json")
        final = rep.get("final") or {}
        frozen = rep.get("frozen") or job.get("frozen") or {}
        g = rep.get("guard") or job.get("guard") or {}
        cards = {k: _card((final.get(k) or {}).get("card"))
                 for k in ("update_up", "update_down", "delete")}
        cards["prompt_baseline"] = _card((rep.get("prompt_baseline") or {}).get("card"))
        items.append({
            "id": d.name, "family": job.get("trait"), "model_id": job.get("model_id"),
            "version": versions[key], "verdict": rep.get("verdict", job.get("verdict")),
            "layer": frozen.get("layer"), "dose": frozen.get("dose"),
            "created": job.get("created"),
            "cards": {k: v for k, v in cards.items() if v is not None},
            "guard": {"blocked": bool(g.get("blocked")), "would_block": bool(g.get("would_block")),
                      "verified": bool(g.get("verified"))},
            "patch": f"/api/traits/{d.name}/patch" if (d / "trait" / "knob.lobot").is_file() else None,
        })
    items.sort(key=lambda t: t["created"] or 0, reverse=True)
    return items


def trait_file(jobs_dir: Path, job_id: str, name: str) -> Path | None:
    """trait/<name> for a finished job, or None. name is one of the served files."""
    assert name in ("report.json", "report.md", "knob.lobot")
    d = job_dir(jobs_dir, job_id)
    if d is None or _read(d / "job.json").get("status") != "done":
        return None
    p = d / "trait" / name
    return p if p.is_file() else None


def patch_filename(jobs_dir: Path, job_id: str) -> str:
    """"<family>-v<version>.lobot" for the download header."""
    for t in list_traits(jobs_dir):
        if t["id"] == job_id:
            return f"{t['family']}-v{t['version']}.lobot"
    return f"{job_id}.lobot"
