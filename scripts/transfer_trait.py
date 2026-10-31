"""Carry a finished trait to another model (1000X XM).

Re-derives the trait on --to with the same seed, splits and judge as
the source run, then writes a transfer record with evidence on both
models (transferred, or unavailable with the failed gate).

Needs the model lane (loads the target model).

Run:
    uv run python scripts/transfer_trait.py --source logs/clane_pirate_llama \
        --to Qwen/Qwen3.5-4B --lane llama
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import prompt_bank  # noqa: E402
from model_lobotomy import trait_eval  # noqa: E402
from model_lobotomy import transfer  # noqa: E402
from model_lobotomy.engine import get_engine  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="source run dir (report.json, knob.lobot)")
    ap.add_argument("--to", required=True, help="target model id")
    ap.add_argument("--lane", default="torch", choices=["torch", "llama"])
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args(argv)
    src_dir = Path(args.source)
    src = json.loads((src_dir / "report.json").read_text())
    trait, seed = src["trait"], int(src["seed"])
    if (src.get("fingerprint") or {}).get("model_id") == args.to:
        print("source and target are the same model", flush=True)
        return 2
    cfg = trait_eval.family_cfg(trait)
    eng = get_engine()
    eng.load(args.to)
    lane = None
    if args.lane == "llama":
        from model_lobotomy import llama_backend
        lane = trait_eval.LlamaLane(eng, eng.spec, llama_backend.CreateDaemon(
            args.to, eng.revision, eng.tokenizer))
    workdir = Path(args.workdir or (ROOT / "logs" / f"xm_{trait}_"
                                    f"{args.to.split('/')[-1].lower()}"))
    out = trait_eval.run_trait(eng, eng.spec, trait,
                               prompt_bank.PROMPT_BANK, seed, workdir,
                               args.to, eng.revision,
                               schedule=src.get("schedule", "full"),
                               lane=lane)
    rec = transfer.transfer_record(
        trait, cfg["judge"], src, out,
        source_patch=src_dir / "knob.lobot",
        target_patch=workdir / "knob.lobot")
    path = transfer.save_record(workdir / "transfer.json", rec)
    print(f"[transfer] {trait}: {rec['status']} "
          f"({rec['target']['unavailable'] or 'gates pass'}) -> {path}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
