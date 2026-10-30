"""Per-family judge validation on hand-written probe pairs (O1/O3).

Scores each probe pair (question, hand-written trait reply,
hand-written anti-trait reply) with the frozen judge. A pair counts
correct when the trait reply scores >= 0.5 and the anti-trait reply
scores < 0.5. Writes logs/judge_<trait>.json and, when agreement meets
--min-agreement, marks the family validated in the trait store
(required before Create accepts the trait).

The 0.80 default is an operator choice recorded in the validation
note, not a plan threshold (the plan sets none).

Run:
    uv run python scripts/validate_judge.py --trait blunt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine as engine_mod  # noqa: E402
import steer  # noqa: E402
import trait_eval  # noqa: E402
import trait_store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", required=True)
    ap.add_argument("--model", default=steer.LLAMA_MODEL_ID)
    ap.add_argument("--min-agreement", type=float, default=0.80)
    ap.add_argument("--store", default=str(trait_store.DEFAULT_PATH))
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()
    cfg = dict(trait_eval.spike.BEHAVIORS[args.trait])
    workdir = Path(args.workdir or (ROOT / "logs" / f"judge_{args.trait}"))
    workdir.mkdir(parents=True, exist_ok=True)
    eng = engine_mod.Engine()
    eng.load(args.model)
    print(f"[judge] {args.model} warm (rev {eng.revision[:12]})",
          flush=True)
    try:
        spec = eng.spec
        pairs = []
        correct = 0
        for q, pos_r, neg_r in cfg["probe"]:
            jp = trait_eval.judge_score(
                eng.model, eng.tokenizer, spec, q, pos_r, cfg["judge"])
            jn = trait_eval.judge_score(
                eng.model, eng.tokenizer, spec, q, neg_r, cfg["judge"])
            ok = jp >= 0.5 > jn
            correct += int(jp >= 0.5) + int(jn < 0.5)
            pairs.append({"q": q, "pos": round(jp, 3),
                          "neg": round(jn, 3), "correct": ok})
        n_labels = 2 * len(pairs)
        agreement = correct / n_labels
        passed = agreement >= args.min_agreement
        out = {"trait": args.trait, "n_labels": n_labels,
               "agreement": round(agreement, 3),
               "min_agreement": args.min_agreement,
               "passed": passed, "pairs": pairs,
               "fingerprint": eng.fingerprint()}
        (workdir / "report.json").write_text(json.dumps(out, indent=2))
        store = trait_store.TraitStore(path=args.store)
        if passed:
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
        return 0 if passed else 1
    finally:
        eng.unload()


if __name__ == "__main__":
    raise SystemExit(main())
