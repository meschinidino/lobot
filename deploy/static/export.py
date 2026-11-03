"""Export the replay server's answers as static files (read-only demo).

Runs the FastAPI app in process with replay mode and the public-demo rules,
calls every read route the page uses, and writes one JSON file per answer.
static-api.js serves these files in the browser, so a static host needs no
server. Muse Spark is never called: explain and second opinion return
saved answers only.

Usage (repo root): uv run python deploy/static/export.py <site-dir>
Answers go to <site-dir>/static-api/; patch files go to their own paths.
LOBOT_JOBS_DIR picks the jobs, as for the server.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

os.environ.pop("META_API_KEY", None)
os.environ["MODEL_LOBOTOMY_REPLAY"] = "1"
os.environ["LOBOT_PUBLIC_DEMO"] = "1"
os.environ["MODEL_LOBOTOMY_REPLAY_STAGE_S"] = "0"

from fastapi.testclient import TestClient  # noqa: E402

from model_lobotomy import app as engine  # noqa: E402
from model_lobotomy import jobs_store  # noqa: E402
from model_lobotomy import server  # noqa: E402
from model_lobotomy import server_helpers as helpers  # noqa: E402

START_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def fnv1a(text: str) -> str:
    """32-bit FNV-1a, hex. static-api.js uses the same hash for file names."""
    h = 0x811C9DC5
    for b in text.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def main(site: Path) -> None:
    out = site / "static-api"
    out.mkdir(parents=True, exist_ok=True)
    c = TestClient(server.api)
    index: dict[str, str] = {}

    # Job records hold absolute paths (workdirs, tracebacks). Publish none.
    local = [(str(jobs_store.default_jobs_dir().resolve().parent), "<repo>"),
             (str(Path.home()), "~")]

    def save(key: str, status: int, body) -> None:
        name = fnv1a(key)
        text = json.dumps({"status": status, "body": body})
        for path, alias in local:
            text = text.replace(path, alias)
        (out / f"{name}.json").write_text(text)
        index[key] = name

    def get(path: str):
        r = c.get(path)
        try:
            body = r.json()
        except ValueError:
            return None
        save(f"GET {path}", r.status_code, body)
        return body if r.is_success else None

    def post(path: str) -> bool:
        r = c.post(path, json={})
        save(f"POST {path}", r.status_code, r.json())
        return r.is_success

    models = sorted(engine._PRESETS)
    states = {}
    for m in models:
        engine._active_model_id = m
        st = c.get("/api/state").json()
        st.update(static=True, public_demo=True)
        states[m] = st
    for path in ("/api/families", "/api/jobs", "/api/agents", "/api/install/targets",
                 "/api/combine/installed", "/api/features"):
        get(path)

    trait_ids, shelf = set(), {}
    for m in models:
        engine._active_model_id = m
        # The page encodes the model id, so the key does too.
        got = get(f"/api/traits?model_id={quote(m, safe='')}") or {}
        for t in got.get("traits", []):
            trait_ids.add(t["id"])
            shelf[t["id"]] = {"id": t["id"], "family": t["family"], "version": t["version"], "model_id": m}
            if t.get("patch"):
                # A download link, not a fetch: write the file at its own path.
                dest = site / t["patch"].lstrip("/")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(c.get("/" + t["patch"].lstrip("/")).content)
    engine._active_model_id = START_MODEL
    explained = []
    for tid in sorted(trait_ids):
        get(f"/api/traits/{tid}/report")
        get(f"/api/traits/{tid}/pareto")
        has_explain = post(f"/api/traits/{tid}/explain")
        has_opinion = post(f"/api/traits/{tid}/second-opinion")
        if has_explain and has_opinion:
            explained.append(shelf[tid])
    for job in (get("/api/jobs") or {}).get("jobs", []):
        get(f"/api/create/{job['id']}")
    for agent in (get("/api/agents") or {}).get("agents", []):
        get(f"/api/agent/{agent['id']}")

    recordings = helpers._recordings()
    neurons = {(r["layer"], r["neuron"]) for rec in recordings for r in rec["rows"]}
    for m in models:
        engine._active_model_id = m
        for layer, neuron in sorted(neurons):
            r = c.get(f"/api/neuron/{layer}/{neuron}?k=12")
            if r.is_success:
                save(f"GET {m} /api/neuron/{layer}/{neuron}?k=12", r.status_code, r.json())
    find = c.get("/api/find?word=storm&k=8")
    # The demo guide (demo-guide.js) lists only what this export can play.
    guide = {"explained": explained}
    meta = {"start_model": START_MODEL, "states": states, "recordings": recordings,
            "find": {"status": find.status_code, "body": find.json()}, "guide": guide}
    text = json.dumps(meta)
    for path, alias in local:
        text = text.replace(path, alias)
    (out / "meta.json").write_text(text)
    (out / "index.json").write_text(json.dumps(index, indent=0, sort_keys=True))
    print(f"exported {len(index)} answers, {len(recordings)} recordings, "
          f"{len(trait_ids)} behaviors to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
