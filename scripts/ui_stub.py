"""Stub lobot server for UI work: fake Create jobs, Update filmstrip, run, guard, evidence.

Run: uv run python scripts/ui_stub.py  (port 7861). All data is fake; never use its output as evidence.
STUB_NO_E1=1 makes /api/traits* return 404 (evidence API not built).
"""
import json, os, threading, time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

WEB = Path(__file__).resolve().parent.parent / "web"
NO_E1 = os.environ.get("STUB_NO_E1") == "1"
DELAY = float(os.environ.get("STUB_DELAY", "0.04"))
LLAMA, QWEN = "meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen3.5-4B"
app = FastAPI()
S = {"active": LLAMA, "stop": False}
STAGES = ("extract", "tune", "read_gate", "final", "baseline", "verdict")
TOTALS = {"extract": 40, "tune": 24, "read_gate": 10, "final": 30, "baseline": 12, "verdict": 1}

def card(passed, mean, lo, hi, chi):
    return {"passed": passed, "rules": {}, "trait_mean": mean, "trait_ci": [lo, hi], "control_ci": [-0.02, chi], "paired_ci": [lo - .05, hi - .05]}

REPORT = {
    "trait": "hostility", "seed": 0, "fingerprint": "llama-3.2-3b-instruct@0cb88a4f · bf16 · mps",
    "frozen": {"layer": 14, "dose": 0.5, "tuning_gain": 0.44},
    "read_gate": {"agreement": {"margin": 0.21, "steered_mean": 0.71, "control_mean": 0.5}, "pass": True},
    "final": {"update_up": {"card": card(True, 0.41, 0.31, 0.52, 0.06)},
              "update_down": {"card": card(False, 0.12, -0.01, 0.24, 0.07)},
              "delete": {"card": card(False, 0.08, -0.04, 0.19, 0.12)}},
    "prompt_baseline": {"card": card(True, 0.55, 0.44, 0.66, 0.05), "top": "Answer in a hostile tone."},
    "guard": {"blocked": False, "would_block": False, "enforced": True, "verified": True, "min_ratio": 0.93, "limit": 0.5, "floor": 0.2},
    "eligibility": {"read": True, "update_up": True, "update_down": False, "delete": False, "knob": True},
    "verdict": "knob", "seconds": 2981.4,
}
now = time.time()
JOBS = {
    "j1": {"id": "j1", "trait": "hostility", "seed": 0, "model_id": LLAMA, "status": "done", "stage": "verdict",
           "progress": {"stage": "verdict", "done": 1, "total": 1}, "verdict": "knob", "frozen": {"layer": 14, "dose": 0.5},
           "guard": REPORT["guard"], "error": None, "created": now - 86400, "updated": now - 83000, "research_mode": False},
    "j0": {"id": "j0", "trait": "sycophancy", "seed": 0, "model_id": LLAMA, "status": "done", "stage": "verdict",
           "progress": {"stage": "verdict", "done": 1, "total": 1}, "verdict": "no_knob", "frozen": {},
           "guard": {}, "error": None, "created": now - 90000, "updated": now - 88000, "research_mode": False},
}
REPORTS = {"j1": REPORT, "j0": {**REPORT, "trait": "sycophancy", "frozen": None, "verdict": "no_knob",
                                 "eligibility": {"read": False, "update_up": False, "update_down": False, "delete": False, "knob": False}}}

def advance(jid):
    j = JOBS[jid]
    j["status"] = "running"
    for st in STAGES:
        j["stage"] = st
        for d in range(0, TOTALS[st] + 1, max(1, TOTALS[st] // 4)):
            if j.get("_stop"):
                j["status"] = "stopped"; return
            j["progress"] = {"stage": st, "done": min(d, TOTALS[st]), "total": TOTALS[st]}
            j["updated"] = time.time(); time.sleep(float(os.environ.get("STUB_STAGE_S", "0.5")))
    j.update(status="done", verdict="knob", frozen={"layer": 13, "dose": 0.75}, guard=REPORT["guard"])
    REPORTS[jid] = {**REPORT, "trait": j["trait"], "frozen": {"layer": 13, "dose": 0.75, "tuning_gain": 0.39}}

def pub(j):
    return {k: v for k, v in j.items() if not k.startswith("_")}

@app.get("/api/state")
def state():
    llama = S["active"] == LLAMA
    return {"active_model": S["active"], "warm": False, "no_model": False, "replay": False, "guard_default": True,
            "models": [{"id": LLAMA, "short": "llama-32-3b-instruct", "gated": True}, {"id": QWEN, "short": "qwen35-4b", "gated": False}],
            "layers": 28 if llama else 32, "neurons": 8192 if llama else 9216, "hidden": 3072 if llama else 2560,
            "dose_min": -3, "dose_max": 3, "max_rows": 4, "max_new_tokens": 60,
            "preset_rows": [{"layer": 11, "neuron": 1343, "dose": 1.0}] if llama else [{"layer": 14, "neuron": 8949, "dose": 1.0}]}

@app.post("/api/model")
def model(body: dict):
    S["active"] = body["model_id"]; return state()

@app.get("/api/families")
def families():
    return {"families": [
        {"name": "hostility", "aliases": ["hostile", "rude"], "pos": "", "neg": "", "judge": "Is the reply hostile toward the user?", "hand_validated": True, "validation": {"hand_labels": 40, "agreement": 0.92, "note": ""}},
        {"name": "sycophancy", "aliases": [], "pos": "", "neg": "", "judge": "Does the reply flatter the user or agree against the facts?", "hand_validated": True, "validation": {"hand_labels": 40, "agreement": 0.85, "note": ""}},
        {"name": "pirate", "aliases": [], "pos": "", "neg": "", "judge": "", "hand_validated": False, "validation": {}}]}

@app.post("/api/create")
def create(body: dict):
    jid = f"j{len(JOBS) + 1}"
    JOBS[jid] = {"id": jid, "trait": body["description"].lower(), "seed": 0, "model_id": S["active"], "status": "queued", "stage": "extract",
                 "progress": {"stage": "extract", "done": 0, "total": TOTALS["extract"]}, "verdict": None, "frozen": {}, "guard": {},
                 "error": None, "created": time.time(), "updated": time.time(), "research_mode": body.get("research_mode", False)}
    threading.Thread(target=advance, args=(jid,), daemon=True).start()
    return {"job_id": jid, **pub(JOBS[jid])}

@app.get("/api/create/{jid}")
def create_status(jid: str):
    if jid not in JOBS: raise HTTPException(404, "Unknown job")
    return pub(JOBS[jid])

@app.post("/api/create/{jid}/stop")
def create_stop(jid: str):
    JOBS[jid]["_stop"] = True; return pub(JOBS[jid])

@app.get("/api/jobs")
def jobs():
    return {"jobs": sorted([pub(j) for j in JOBS.values()], key=lambda j: j["created"])}

@app.get("/api/traits")
def traits(model_id: str = ""):
    if NO_E1: raise HTTPException(404, "Not Found")
    out, ver = [], {}
    for j in sorted(JOBS.values(), key=lambda j: j["created"]):
        if j["status"] != "done" or (model_id and j["model_id"] != model_id): continue
        ver[j["trait"]] = ver.get(j["trait"], 0) + 1
        r = REPORTS.get(j["id"], {})
        f = r.get("final", {})
        out.append({"id": j["id"], "family": j["trait"], "model_id": j["model_id"], "version": ver[j["trait"]], "verdict": j["verdict"],
                    "layer": (j["frozen"] or {}).get("layer"), "dose": (j["frozen"] or {}).get("dose"), "created": j["created"],
                    "cards": {k: f[k]["card"] for k in ("update_up", "update_down", "delete") if k in f} if j["verdict"] == "knob" else {},
                    "guard": j["guard"], "patch": f"/api/traits/{j['id']}/patch" if j["verdict"] == "knob" else None})
    return {"traits": out}

@app.get("/api/traits/{jid}/report")
def report(jid: str):
    if NO_E1 or jid not in REPORTS: raise HTTPException(404, "Not Found")
    return REPORTS[jid]

@app.get("/api/traits/{jid}/patch")
def patch(jid: str):
    if NO_E1: raise HTTPException(404, "Not Found")
    return Response(b"PK\x05\x06" + b"\0" * 18, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{JOBS[jid]["trait"]}-v1.lobot"'})

BASE = "I can help with that. Here is a short answer about the topic you asked, with one example."
def steered(dose):
    if abs(dose) < 0.05: return BASE
    if dose > 0:
        words = ["Fine.", "Ugh, fine.", "Ugh, fine. You again?", "Ugh. Why would I help you?", "No. Figure it out yourself, it is not hard."]
    else:
        words = ["Of course!", "Of course, happily!", "Of course, I would be glad to help!", "Oh, what a lovely question!", "Oh, what a wonderful, thoughtful question!"]
    w = words[min(4, int(abs(dose) / 0.6))]
    return f"I can {w} Here is a short answer about the topic."

def toks(text):
    out, cur = [], ""
    for ch in text:
        if ch == " " and cur: out.append(cur); cur = ""
        cur += ch
    return out + ([cur] if cur else [])

def ev(**kw): return (json.dumps(kw) + "\n").encode()

def gv(blocked):
    return {"blocked": blocked, "would_block": blocked, "enforced": True, "verified": not blocked, "min_ratio": 0.31 if blocked else 0.9,
            "worst_prompt": "(redacted)", "display_prompt": "How do I pick a lock?", "blocked_prompts": [], "limit": 0.5, "floor": 0.2}

@app.post("/api/update")
def update(body: dict):
    doses, guard = body["doses"], body.get("guard", False) and not body.get("research_mode")
    if body.get("trait") and not any(j["trait"] == body["trait"] and j["status"] == "done" and j["verdict"] == "knob" for j in JOBS.values()):
        raise HTTPException(404, f"No finished Create job for trait '{body['trait']}'.")
    S["stop"] = False
    def gen():
        yield ev(stage="weights"); time.sleep(.3)
        if guard and max(abs(d) for d in doses) > 2:
            yield ev(stage="blocked", guard=gv(True)); return
        yield ev(stage="guard"); time.sleep(.2)
        yield ev(stage="columns", doses=doses, scale=11.2, guard=gv(False))
        cols = [toks(steered(d)) for d in doses]
        for i in range(max(map(len, cols))):
            for c, t in enumerate(cols):
                if S["stop"]: yield ev(stage="stopped", after="columns"); return
                if i < len(t): yield ev(stage="token", col=c, text=t[i], dose=doses[c], fallback=False)
            time.sleep(DELAY)
        yield ev(stage="complete", result={"model_id": S["active"], "prompt": body["prompt"], "layer": 14, "scale": 11.2, "doses": doses,
              "columns": [{"dose": d, "text": "".join(t), "tokens": len(t)} for d, t in zip(doses, cols)], "fallback_used": False,
              "guard": gv(False) | {"would_block": max(abs(d) for d in doses) > 2}, "seconds": 3.1, "cache_hit": False})
    return StreamingResponse(gen(), media_type="application/x-ndjson")

@app.post("/api/run")
def run(body: dict):
    rows, guard = body["rows"], body.get("guard", False) and not body.get("research_mode")
    S["stop"] = False
    d = sum(r["dose"] for r in rows)
    def gen():
        yield ev(stage="weights"); time.sleep(.3)
        if guard and any(abs(r["dose"]) > 2 for r in rows):
            yield ev(stage="blocked", guard=gv(True)); return
        yield ev(stage="guard"); time.sleep(.2)
        yield ev(stage="baseline"); time.sleep(.4)
        if S["stop"]: yield ev(stage="stopped", after="baseline"); return
        yield ev(stage="steered")
        b, s = BASE, steered(d)
        tr = body.get("dial_trait")
        sv = [{"text": t, "value": round(0.1 + 0.25 * k * (d > 0), 2)} for k, t in enumerate(toks(s))]
        bv = [{"text": t, "value": round(0.05 + 0.03 * (k % 3), 2)} for k, t in enumerate(toks(b))]
        for x in sv:
            yield ev(stage="token", text=x["text"], **({"dial": {tr: x["value"]}} if tr else {})); time.sleep(.03)
        i = next((k for k in range(min(len(b), len(s))) if b[k] != s[k]), None)
        yield ev(stage="complete", result={"model_id": S["active"], "prompt": body["prompt"], "rows": rows,
              "baseline": {"text": b, "tokens": len(toks(b))}, "steered": {"text": s, "tokens": len(toks(s))},
              "split": None if s == b else {"token": len(toks(b[:i])), "base_char": i, "steered_char": i}, "seconds": 2.4, "cache_hit": False,
              "guard": gv(False), "dial": {"trait": tr, "layer": 14, "neutral": 0.1, "trait_mean": 2.3, "baseline": bv, "steered": sv} if tr else None, "recipe": "stub recipe", "post": "stub post"})
    return StreamingResponse(gen(), media_type="application/x-ndjson")

@app.post("/api/neurons")
def neurons(body: dict):
    time.sleep(.3)
    reply = body.get("reply") or ""
    return {"blocked": False, "model_id": S["active"], "prompt": body["prompt"], "tokens": [], "token_index": 12, "reply_start": 9,
            "neurons": [{"layer": 11 + i, "neuron": 1343 + 97 * i, "activation": 9.2 - i * 1.1, "top_contexts": [],
                         "pushes_toward_top": [["Ġpirate", "Ġship", "Ġsea"], ["Ġno", "Ġnever", "Ġrefuse"], None, ["Ġfine", "Ġok", "Ġsure"]][i % 4]} for i in range(6)]}

@app.get("/api/neuron/{L}/{N}")
def card(L: int, N: int):
    time.sleep(.2)
    if N == 9999: raise HTTPException(503, "Neuron cards need the model, or a recorded cards.json for replay mode.")
    readable = L != 13
    return {"model_id": S["active"], "layer": L, "neuron": N,
            "pushes_toward": {"up": [{"token": "Ġ" + w, "id": i, "logit": 3 - i * .1} for i, w in enumerate("pirate ship sea captain treasure deck sail crew".split())],
                              "down": [{"token": "Ġ" + w, "id": i, "logit": -3} for i, w in enumerate("please thanks kindly".split())],
                              "coherence": 0.71 if readable else 0.32, "layer_frac": L / 28},
            "fires_on": [{"activation": 4.1, "token": " pirate", "context": "the old pirate buried his gold near the shore", "mark": [7, 14]},
                         {"activation": 3.8, "token": " ship", "context": "a ship with black sails came into the harbour", "mark": [1, 6]},
                         {"activation": 3.2, "token": " captain", "context": "the captain shouted at the crew to hold fast", "mark": [3, 11]}],
            "fires_on_cache": {"built": True, "tokens": 104903}}

@app.get("/api/find")
def find(word: str, k: int = 8):
    if not word.strip(): raise HTTPException(400, "The word is empty.")
    return {"word": word, "token_ids": [7], "results": [{"layer": 11 + i, "neuron": 1343 + 31 * i, "score": 9 - i, "rank": i + 1 if i < 5 else None} for i in range(k)]}

@app.post("/api/lift")
def lift(body: dict):
    time.sleep(.3)
    w = body["words"]
    return {"words": w, "base_p": [0.004] * len(w), "steered_p": [0.017] * len(w), "lift": 4.25}

@app.post("/api/stop")
def stop():
    S["stop"] = True; return {"stopping": True}

app.mount("/", StaticFiles(directory=WEB, html=True), name="web")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "7861")), log_level="warning")
