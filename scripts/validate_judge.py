"""Per-family judge validation on hand-written probe pairs (O1/O3).

Scores each probe pair (question, hand-written trait reply,
hand-written anti-trait reply) with the frozen judge. A pair counts
correct when the trait reply scores >= 0.5 and the anti-trait reply
scores < 0.5. Writes logs/judge_<trait>.json and, when agreement meets
--min-agreement, marks the family validated in the trait store
(required before Create accepts the trait).

The 0.80 default is an operator choice recorded in the validation
note, not a plan threshold (the plan sets none).

--judge meta_api scores the same pairs with the Meta model API
(J1, offline only, no model load). It never marks the family
validated: the local judge gates Create. It writes
logs/judge_<trait>_meta_api.json and never logs the key.

Run:
    uv run python scripts/validate_judge.py --trait blunt
    META_API_KEY=... LOBOT_JUDGE_MODEL=... \
        uv run python scripts/validate_judge.py --trait blunt \
        --judge meta_api
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import engine as engine_mod  # noqa: E402
from model_lobotomy import steer  # noqa: E402
from model_lobotomy import trait_eval  # noqa: E402
from model_lobotomy import trait_store  # noqa: E402

META_WORKERS = 4


def _tally(scored):
    """Agreement counts from a [(q, pos_score, neg_score), ...] list."""
    pairs, correct = [], 0
    for q, jp, jn in scored:
        ok = jp >= 0.5 > jn
        correct += int(jp >= 0.5) + int(jn < 0.5)
        pairs.append({"q": q, "pos": round(jp, 3), "neg": round(jn, 3),
                      "correct": ok})
    n_labels = 2 * len(pairs)
    return pairs, correct / n_labels, n_labels


def _write_and_judge(args, workdir, pairs, agreement, n_labels,
                     judge_block, fingerprint, usage):
    out = {"trait": args.trait, "n_labels": n_labels,
           "agreement": round(agreement, 3),
           "min_agreement": args.min_agreement,
           "passed": agreement >= args.min_agreement, "pairs": pairs,
           "judge": judge_block, "fingerprint": fingerprint,
           "usage": usage}
    (workdir / "report.json").write_text(json.dumps(out, indent=2))
    return out


def _local_main(args, cfg, workdir) -> int:
    eng = engine_mod.Engine()
    eng.load(args.model)
    print(f"[judge] {args.model} warm (rev {eng.revision[:12]})",
          flush=True)
    try:
        spec = eng.spec
        scored = [
            (q, trait_eval.judge_score(
                eng.model, eng.tokenizer, spec, q, pos_r, cfg["judge"]),
             trait_eval.judge_score(
                 eng.model, eng.tokenizer, spec, q, neg_r, cfg["judge"]))
            for q, pos_r, neg_r in cfg["probe"]]
        pairs, agreement, n_labels = _tally(scored)
        out = _write_and_judge(
            args, workdir, pairs, agreement, n_labels,
            trait_eval.judge_info("local", args.model,
                                  round(agreement, 3)),
            eng.fingerprint(), None)
        store = trait_store.TraitStore(path=args.store)
        if out["passed"]:
            store.mark_validated(
                args.trait, n_labels=n_labels, agreement=agreement,
                note=f"O1/O3 probe validation, min {args.min_agreement}")
            store.save()
            print(f"[judge] {args.trait}: agreement {agreement:.3f} "
                  f"-> VALIDATED", flush=True)
        else:
            print(f"[judge] {args.trait}: agreement {agreement:.3f} "
                  f"-> NOT validated (min {args.min_agreement})",
                  flush=True)
        return 0 if out["passed"] else 1
    finally:
        eng.unload()


def _meta_main(args, cfg, workdir) -> int:
    judge = trait_eval.MetaApiJudge()

    def one(triple):
        q, pos_r, neg_r = triple
        return (q, judge.score(q, pos_r, cfg["judge"]),
                judge.score(q, neg_r, cfg["judge"]))

    with ThreadPoolExecutor(max_workers=META_WORKERS) as pool:
        scored = list(pool.map(one, cfg["probe"]))
    pairs, agreement, n_labels = _tally(scored)
    out = _write_and_judge(
        args, workdir, pairs, agreement, n_labels,
        trait_eval.judge_info("meta_api", judge.model,
                              round(agreement, 3)),
        None, judge.usage)
    print(f"[judge] {args.trait}: agreement {agreement:.3f} "
          f"(meta_api {judge.model}; not stored, "
          f"local judge gates Create)", flush=True)
    print(f"[judge] usage: {judge.usage['calls']} calls, "
          f"{judge.usage['input_tokens']} in / "
          f"{judge.usage['output_tokens']} out tokens", flush=True)
    return 0 if out["passed"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", required=True)
    ap.add_argument("--model", default=steer.LLAMA_MODEL_ID)
    ap.add_argument("--judge", choices=("local", "meta_api"),
                    default="local")
    ap.add_argument("--min-agreement", type=float, default=0.80)
    ap.add_argument("--store", default=str(trait_store.DEFAULT_PATH))
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()
    cfg = dict(trait_eval.spike.BEHAVIORS[args.trait])
    dirname = (f"judge_{args.trait}_meta_api" if args.judge == "meta_api"
               else f"judge_{args.trait}")
    workdir = Path(args.workdir or (ROOT / "logs" / dirname))
    workdir.mkdir(parents=True, exist_ok=True)
    if args.judge == "meta_api":
        return _meta_main(args, cfg, workdir)
    return _local_main(args, cfg, workdir)


if __name__ == "__main__":
    raise SystemExit(main())
