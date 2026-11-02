"""Browser walk of every screen in replay mode. No weights.

Starts the replay server, switches it to Llama, then walks:
the workbench (and one recorded run), the share card, chat,
the Library, every evidence page (and steps its dose dial),
New behavior, every Muse run and every labeling page.
It fails on any pageerror. It writes the page text to --out.
Run the same script on main and on a branch, then diff the outs.

Run:
    .venv/bin/python scripts/ui_walk.py --out /tmp/ui_walk_branch.txt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:7860"
LLAMA = "meta-llama/Llama-3.2-3B-Instruct"


def _get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read())


def _post(path, body):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _wait_server(proc, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"server exited with {proc.returncode}; "
                             "see /tmp/ui_walk_server.log")
        try:
            _get("/api/state")
            return
        except Exception:
            time.sleep(1)
    raise SystemExit("server did not answer /api/state in time")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/ui_walk.txt")
    args = ap.parse_args()

    env = {**os.environ, "MODEL_LOBOTOMY_REPLAY": "1"}
    log = open("/tmp/ui_walk_server.log", "wb")
    proc = subprocess.Popen(
        [sys.executable, "-m", "model_lobotomy.server"],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL)
    chunks: list[str] = []
    errors: list[str] = []
    try:
        _wait_server(proc)
        _post("/api/model", {"model_id": LLAMA})

        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))

            def snap(name):
                page.wait_for_timeout(400)
                text = page.evaluate(
                    "document.body ? document.body.innerText : ''")
                chunks.append(f"\n===== {name} =====\n{text}\n")

            def open_page(kind, id=None):
                page.evaluate(
                    """async ([kind, id]) => {
                      const m = await import('/js/screens/pages.js');
                      m.openPage(id ? {kind, id} : {kind});
                    }""", [kind, id])
                page.wait_for_timeout(700)

            page.goto(BASE + "/", wait_until="networkidle")
            snap("workbench")

            # One recorded run on the workbench.
            page.click("#runBtn")
            page.wait_for_function(
                "() => { const b = document.querySelector('#shareBtn');"
                " return b && !b.disabled; }",
                timeout=60000)
            snap("workbench-run")

            page.click("#shareBtn")
            snap("share-card")

            page.evaluate("window.scrollTo(0, 0)")
            open_page("chat")
            snap("chat")

            open_page("library")
            snap("library")

            traits = _get("/api/traits").get("traits", [])
            for item in traits:
                tid = item["id"]
                open_page("behavior", tid)
                snap(f"evidence:{tid}")
                dial = page.query_selector("#exDial")
                if dial is not None:
                    dial.focus()
                    before = page.evaluate(
                        "document.querySelector('#exOut')?.innerText || ''")
                    page.keyboard.press("ArrowRight")
                    page.wait_for_timeout(400)
                    after = page.evaluate(
                        "document.querySelector('#exOut')?.innerText || ''")
                    chunks.append(
                        f"\n===== evidence-dial:{tid} =====\n"
                        f"changed={before != after}\n{after}\n")

            open_page("new")
            snap("new-behavior")

            agents = _get("/api/agents").get("agents", [])
            for run in agents:
                open_page("agent", run["run_id"])
                snap(f"agent:{run['run_id']}")

            drafts_dir = ROOT / "store" / "drafts"
            draft_ids = sorted(p.name for p in drafts_dir.iterdir()) \
                if drafts_dir.is_dir() else []
            for did in draft_ids:
                open_page("draft", did)
                snap(f"labeling:{did}")

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    Path(args.out).write_text("".join(chunks))
    print(f"wrote {args.out}: {len(chunks)} screens, "
          f"{len(errors)} pageerrors")
    for e in errors:
        print(f"pageerror: {e}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
