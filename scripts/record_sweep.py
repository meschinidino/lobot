"""Record one prompt at several doses, for the prototype's experiment tree.

Runs the baseline once, then the same recipe scaled to each dose (for
example x-1, x-0.5, x+0.5, x+1), with the same engine, calibration, seed
and split rule as the web app. Writes web/sweep.json. Then rebuild the
prototype so its tree shows every recorded run:

    uv run python scripts/record_sweep.py [--model Qwen/Qwen3.5-4B]
        [--prompt "..."] [--rows 14:8949:1,13:7956:-1] [--scales -1,-0.5,0.5,1]
    uv run python scripts/build_prototype.py

Qwen writes web/sweep.json (the prototype's tree); other models write
web/sweep.<short name>.json. The server's replay mode and the live app read
them all. The recipe defaults to the model's preset rows. Nothing is posted anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import app as engine
from model_lobotomy import steer
from model_lobotomy.engine import EditSet, Read, get_engine
from model_lobotomy.server import _dial_spec, _split_char
from model_lobotomy.steer import RecipeRow


def parse_rows(text: str) -> list[RecipeRow]:
    rows = []
    for part in text.split(","):
        layer, neuron, dose = part.strip().split(":")
        rows.append(RecipeRow(int(layer), int(neuron), float(dose)))
    return rows


def _run_pass(eng, prompt, edits, medians, max_new, dial_spec):
    """One greedy pass; with dial_spec also its per-token [{"text", "value"}].

    The dial path streams (same greedy run, same seed) so each token
    carries its read-dial value; without it this is one plain Engine.run.
    """
    if dial_spec is None:
        return eng.run(prompt, edits, medians=medians,
                       max_new_tokens=max_new, seed=engine.SEED), []
    values = []
    result = None
    for ev in eng.run_stream(prompt, edits, medians=medians,
                             max_new_tokens=max_new, seed=engine.SEED,
                             dial=dial_spec):
        if ev["type"] == "token":
            d = ev.get("dial") or {}
            values.append({"text": ev["text"],
                           "value": d.get(dial_spec["trait"])})
        else:
            result = ev["result"]
    return result, values


def main() -> int:
    example = json.loads((ROOT / "web" / "example.json").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=steer.MODEL_ID, choices=sorted(engine._PRESETS))
    ap.add_argument("--prompt", default=example["prompt"])
    ap.add_argument("--rows", default=None,
                    help="layer:neuron:dose,... (default: the model's preset rows)")
    ap.add_argument("--scales", default="-1,-0.5,0.5,1",
                    help="dose multipliers for the recipe; the baseline always runs")
    ap.add_argument("--out", default=None,
                    help="default: web/sweep.json for Qwen, web/sweep.<short name>.json for others")
    ap.add_argument("--dial-trait", default=None,
                    help="record per-token read-dial lists for this trait family "
                    "(needs a finished Create job; skipped with a note when missing)")
    args = ap.parse_args()

    if args.out is None:
        name = "sweep.json" if args.model == steer.MODEL_ID else f"sweep.{steer.model_short_name(args.model)}.json"
        args.out = str(ROOT / "web" / name)
    engine._active_model_id = args.model
    preset = engine.active_preset()
    spec = engine.active_spec()
    rows = parse_rows(args.rows) if args.rows else [
        RecipeRow(r["layer"], r["neuron"], r["dose"]) for r in preset["rows"]]
    scales = [float(s) for s in args.scales.split(",") if float(s) != 0]
    for s in scales:
        steer.validate_recipe([RecipeRow(r.layer, r.neuron, r.dose * s) for r in rows], spec)
    prompt = args.prompt.strip()
    if not prompt:
        print("empty prompt", flush=True)
        return 2
    calib_prompts = list(preset["calibration_prompts"])
    max_new = int(preset["decode"]["max_new_tokens"])

    t_all = time.perf_counter()
    # Same engine, calibration and seed as the server's /api/run.
    eng = get_engine()
    if eng.model is None or eng.model_id != args.model:
        eng.load(args.model)
    steer.check_preset_binding(preset, args.model)
    model, tok = eng.model, eng.tokenizer
    dial_spec = None
    if args.dial_trait:
        try:
            dial_spec = _dial_spec(eng, args.dial_trait)
        except KeyError:
            print(f"[sweep] --dial-trait {args.dial_trait}: no finished Create job "
                  f"with a saved direction for that trait. "
                  f"Recording without dial.", flush=True)
    key = steer.calibration_cache_key(
        args.model, eng.revision, tok, str(steer.DTYPE), steer.DEVICE)
    layers = sorted({r.layer for r in rows}
                    | ({dial_spec["layer"]} if dial_spec else set()))
    with eng.clean():
        medians, cal_info = steer.calibrate_layer_medians(
            model, tok, calib_prompts, layers,
            cache_path=ROOT / steer.calibration_path_for(args.model), key=key, spec=spec)

    read = (Read(layer=dial_spec["layer"]),) if dial_spec else ()
    base, base_dial = _run_pass(eng, prompt, EditSet(read=read), medians,
                                max_new, dial_spec)
    print(f"[sweep] baseline: {base['new_tokens']} tokens, {base['seconds']} s", flush=True)
    runs = []
    for s in scales:
        scaled = [RecipeRow(r.layer, r.neuron, round(r.dose * s, 4)) for r in rows]
        t0 = time.perf_counter()
        out, out_dial = _run_pass(
            eng, prompt,
            EditSet(neuron=EditSet.from_rows(scaled).neuron, read=read),
            medians, max_new, dial_spec)
        if out["hook_calls"] < 1:
            print(f"[sweep] x{s:+}: steering hooks did not fire; stopping", flush=True)
            return 4
        div = engine.first_divergence(base["token_ids"], out["token_ids"])
        split = None if div is None else {
            "token": div,
            "base_char": _split_char(tok, base["text"], base["token_ids"], div, out["text"]),
            "steered_char": _split_char(tok, out["text"], out["token_ids"], div, base["text"]),
        }
        kw = dict(rows=scaled, calib_id=cal_info["key"]["prompts_sha256"], max_new_tokens=max_new,
                  seed=engine.SEED, calib_prompts=calib_prompts, model_id=args.model)
        entry = {
            "scale": s,
            "rows": [{"layer": r.layer, "neuron": r.neuron, "dose": r.dose} for r in scaled],
            "text": out["text"], "tokens": out["new_tokens"], "split": split,
            "seconds": round(time.perf_counter() - t0, 1),
            "recipe": steer.recipe_string(**kw),
            "post": steer.build_post_text(base=base, steered=out, **kw),
        }
        if dial_spec is not None:
            entry["dial"] = {"trait": dial_spec["trait"],
                             "layer": dial_spec["layer"],
                             "neutral": dial_spec["b0"],
                             "trait_mean": dial_spec["b1"],
                             "baseline": base_dial, "steered": out_dial}
        runs.append(entry)
        print(f"[sweep] x{s:+}: split at token {div}, {out['new_tokens']} tokens", flush=True)

    result = {
        "model_id": args.model, "prompt": prompt, "seed": engine.SEED,
        "max_new_tokens": max_new, "calib_id": cal_info["key"]["prompts_sha256"],
        "calib_prompts": calib_prompts,
        "rows": [{"layer": r.layer, "neuron": r.neuron, "dose": r.dose} for r in rows],
        "baseline": {"text": base["text"], "tokens": base["new_tokens"]},
        "runs": runs, "seconds": round(time.perf_counter() - t_all, 1),
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"[sweep] wrote {args.out}. Next: uv run python scripts/build_prototype.py",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
