"""Run the Create agent: words in, tested patch out (Muse Spark + local gates).

Needs META_API_KEY and the model lane (loads weights).

Run:
    uv run python scripts/agent_create.py "talks like an old-west cowboy" --lane llama
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import agent as agent_mod  # noqa: E402
from model_lobotomy import steer  # noqa: E402
from model_lobotomy import trait_eval  # noqa: E402
from model_lobotomy.engine import get_engine  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("description")
    ap.add_argument("--model", default=steer.LLAMA_MODEL_ID)
    ap.add_argument("--lane", default="torch", choices=["torch", "llama"])
    ap.add_argument("--max-rounds", type=int, default=agent_mod.MAX_ROUNDS)
    args = ap.parse_args(argv)
    eng = get_engine()
    eng.load(args.model)
    lane = None
    if args.lane == "llama":
        from model_lobotomy import llama_backend
        lane = trait_eval.LlamaLane(eng, eng.spec, llama_backend.CreateDaemon(
            args.model, eng.revision, eng.tokenizer))

    def progress(kind, row):
        brief = {k: v for k, v in row.items()
                 if k in ("round", "agreement", "verdict", "action",
                          "reason", "family", "description", "text")}
        print(f"[agent] {kind} {json.dumps(brief, default=str)}",
              flush=True)

    out = agent_mod.CreateAgent(eng, lane=lane,
                                max_rounds=args.max_rounds,
                                progress=progress).run(args.description)
    print(f"[agent] {out['status']} after {out['rounds']} round(s); "
          f"muse {out['muse']}", flush=True)
    return 0 if out["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
