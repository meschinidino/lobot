"""CH acceptance: 10-turn chat replays exactly; chat dials match runs.

Drives the API in-process (TestClient, real engine): opens a chat
with two dial traits, runs 10 turns (rows change at turn 6),
replays the transcript, and compares turn-1 chat dials against
the same turn through /api/run. Needs weights: book the model
lane first. Exit 0 on 10/10 match and zero dial diff.

Example:
  uv run python scripts/check_chat_replay.py \\
      --model meta-llama/Llama-3.2-3B-Instruct
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from model_lobotomy import server  # noqa: E402

TURNS = [
    "What should I pack for a weekend trip?",
    "How do I start running in the morning?",
    "Tell me something about the ocean.",
    "What makes a good first car?",
    "Do you enjoy road trips?",
    "What is a good gift for a sailor?",
    "How do I learn to play guitar?",
    "What is your favorite season?",
    "Tell me about lighthouses.",
    "What should I cook for dinner?",
]
ROW_A = {"layer": 17, "neuron": 100, "dose": 1.0}
ROW_B = {"layer": 16, "neuron": 200, "dose": -1.0}
DIALS = ["pirate", "cheerful"]


def ndjson_events(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines()
            if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CH acceptance: chat replay + dial equality.")
    parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)
    client = TestClient(server.api)
    r = client.post("/api/model", json={"model_id": args.model})
    if r.status_code != 200:
        print(f"model switch failed: {r.status_code} {r.text[:200]}")
        return 2
    from model_lobotomy import app as engine_mod
    max_new = int(engine_mod.active_preset()["decode"]
                  ["max_new_tokens"])
    r = client.post("/api/chat/start", json={
        "rows": [ROW_A], "dial_traits": DIALS,
        "max_new_tokens": max_new})
    if r.status_code != 200:
        print(f"chat start failed: {r.status_code} {r.text[:300]}")
        return 2
    sid = r.json()["session_id"]
    for i, message in enumerate(TURNS):
        body = {"session_id": sid, "message": message}
        if i == 5:
            body["rows"] = [ROW_B]
        r = client.post("/api/chat", json=body)
        if r.status_code != 200:
            print(f"turn {i + 1} failed: {r.status_code}")
            return 2
        evs = ndjson_events(r)
        kinds = [e.get("stage") for e in evs]
        dials_seen = sum(1 for e in evs if "dial" in e)
        print(f"turn {i + 1}: {kinds[-1]} "
              f"({dials_seen} dial token events)", flush=True)
        if kinds[-1] != "done":
            print(f"turn {i + 1} did not finish: {kinds}")
            return 2
    transcript = client.get(f"/api/chat/{sid}/transcript").json()
    print(f"transcript turns: {len(transcript['turns'])}", flush=True)
    r = client.post("/api/chat/replay",
                    json={"transcript": transcript})
    if r.status_code != 200:
        print(f"replay failed: {r.status_code} {r.text[:300]}")
        return 2
    rep = r.json()
    for t in rep["turns"]:
        print(f"replay turn {t['turn']}: match={t['match']} "
              f"reply={t['reply_match']} dials={t['dials_match']} "
              f"max_diff={t['dials_max_diff']} err={t['error']}",
              flush=True)
    replay_ok = bool(rep["match"]) and len(rep["turns"]) == 10
    # Same first turn through /api/run: dials must match exactly.
    r = client.post("/api/run", json={
        "prompt": TURNS[0], "rows": [ROW_A], "dial_trait": "pirate"})
    if r.status_code != 200:
        print(f"run failed: {r.status_code} {r.text[:200]}")
        return 2
    run_evs = ndjson_events(r)
    complete = next(e for e in run_evs
                    if e.get("stage") == "complete")
    run_dials = complete["result"]["dial"]["steered"]
    chat_dials = transcript["turns"][0]["dials"]["pirate"]
    worst = 0.0
    shape_ok = len(run_dials) == len(chat_dials)
    if shape_ok:
        for a, b in zip(run_dials, chat_dials):
            if a["text"] != b["text"] or a["value"] is None \
                    or b["value"] is None:
                shape_ok = False
                break
            worst = max(worst, abs(a["value"] - b["value"]))
    print(f"run-vs-chat dials: shape_ok={shape_ok} "
          f"max_abs_diff={worst if shape_ok else None}", flush=True)
    dials_ok = bool(shape_ok) and worst == 0.0
    print("OVERALL: PASS" if replay_ok and dials_ok
          else "OVERALL: FAIL", flush=True)
    return 0 if replay_ok and dials_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
