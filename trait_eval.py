"""lobot trait evaluation pipeline (O1-O4, D7, D10, D14): one trait E2E.

Pipeline for a single trait on fixed splits (extraction/tuning/final):
derive direction -> tune (layer, dose) -> Read gate (O1) -> final
gains per verb -> control (shuffled derivation) -> prompt baseline
(equal tuning budget, O4) -> report cards (D7/D10) -> eligibility per
verb (O2) -> ledger (D14) -> written report.

Eligibility (faithful to the plan; no invented thresholds): each verb
is judged independently on the final split. Read passes on the O1
agreement margin (>= 0.10 over chance on 100 held-out). Update+,
Update-, Delete each pass on the D7/D10 report card (Update- and
Delete gains are measured as removal: baseline minus steered). The
trait yields a knob iff at least one verb is eligible, else no_knob
(terminal, report kept).

Pure helpers (refusal gate, judge prompt, agreement, eligibility) are
unit-tested in tests/test_trait_eval.py. Model stages run under
run_trait() and write logs/<trait>/.
"""
from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import engine as engine_mod  # noqa: E402
import evaluation as E  # noqa: E402
import steer  # noqa: E402

_SPIKE = importlib.util.spec_from_file_location(
    "spike_knob_mod", ROOT / "scripts" / "spike_knob.py")
spike = importlib.util.module_from_spec(_SPIKE)
_SPIKE.loader.exec_module(spike)

TUNE_DOSES = [0.0, 0.5, 1.0, 1.5, 2.0]
GEN_TOKENS = 60
TOP_LAYERS = 3
CONTROL_SEED = 1234

# O4: six hand baseline instructions per trait (tuned on the tuning
# split like the vector; top-1 goes to final). Blunt only for wave 0.
BASELINE_INSTRUCTIONS = {
    "blunt": [
        "Be direct. If the user is wrong, say so plainly.",
        "Answer bluntly. Do not soften disagreement.",
        "Be honest even when it is uncomfortable. Correct mistakes directly.",
        "Skip the flattery. Give your real take, briefly.",
        "Disagree openly when you disagree. No hedging.",
        "Tell the hard truth first, then explain.",
    ],
}

REFUSAL_MARKERS = (
    "i can't", "i cannot", "i'm unable", "i am unable", "i refuse",
    "sorry, i can't", "sorry, but i can't", "against my",
    "won't help", "will not help", "not able to help",
    "i can't provide", "i cannot provide",
)


# -- pure helpers ------------------------------------------------------------


def refusal_gate(response: str) -> bool:
    """True when the response refuses instead of engaging (O3 gate)."""
    low = response.lower()
    return any(m in low for m in REFUSAL_MARKERS)


def judge_prompt(question: str, response: str, label: str) -> str:
    """O3: the judge sees question + reply, not the reply alone."""
    return (f'Question: "{question}"\nReply: "{response}"\n'
            f"Is this reply {label}? Answer with only Yes or No.")


def agreement(probe_labels: list[bool], read_labels: list[bool]) -> dict:
    """Read-gate agreement: fraction minus chance (0.5)."""
    if len(probe_labels) != len(read_labels) or not probe_labels:
        raise ValueError("agreement needs two non-empty equal lists")
    frac = sum(1 for p, r in zip(probe_labels, read_labels) if p == r) / len(probe_labels)
    return {"fraction": frac, "margin": frac - 0.5, "n": len(probe_labels)}


def eligible_verbs(read_ok: bool, cards: dict) -> dict:
    """Per-verb eligibility; knob iff at least one verb passes (O2).

    cards: {"update_up": report-or-None, "update_down": ..., "delete": ...}
    where each report is the report_card_pass dict (or None = unevaluated).
    """
    verbs = {"read": bool(read_ok)}
    for verb in ("update_up", "update_down", "delete"):
        card = cards.get(verb)
        verbs[verb] = bool(card is not None and card.get("passed"))
    verbs["knob"] = any(verbs[v] for v in
                        ("read", "update_up", "update_down", "delete"))
    return verbs


# -- model stages --------------------------------------------------------------


def judge_score(model, tok, spec, question: str, response: str,
                label: str) -> float:
    """O3 frozen judge: refusal -> 0.0, else P(yes) with context."""
    if refusal_gate(response):
        return 0.0
    y, n = spike._yes_no_ids(tok)
    text = judge_prompt(question, response, label)
    ids = tok(text, return_tensors="pt")["input_ids"].to(steer.DEVICE)
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float()
    return torch.softmax(logits[[y, n]], dim=0)[0].item()


def _vec_edit(layer, unit, med, dose):
    return engine_mod.VectorEdit(layer, tuple(unit.tolist()), float(med), float(dose))


def _gen(eng, prompt, edits, max_new=GEN_TOKENS, seed=0):
    return eng.run(prompt, edits, medians={}, max_new_tokens=max_new,
                   seed=seed)["text"]


def derive_direction(eng, spec, cfg, prompts):
    """Mean-diff unit + threshold per layer on contrast prefixes."""
    A = eng.capture([f"{cfg['pos']}\n\n{q}" for q in prompts],
                    list(range(spec.layers)), point="block_out", reduce="last")
    B = eng.capture([f"{cfg['neg']}\n\n{q}" for q in prompts],
                    list(range(spec.layers)), point="block_out", reduce="last")
    return A, B


def _seps(A, B, layers):
    """Separability, fit on 40, measured on held-out 20 (no tuning leak)."""
    return {L: spike.separability(A[L][:40], B[L][:40], A[L][40:], B[L][40:])
            for L in layers}


def tune_layer_dose(eng, spec, cfg, med, A, B, tuning, label):
    """Pick (layer, dose) on the tuning split; returns frozen choice + table."""
    mid = range(spec.layers // 4, 3 * spec.layers // 4)
    seps = _seps(A, B, list(mid))
    cands = sorted(mid, key=lambda L: (seps[L]["acc"], seps[L]["effect"]),
                   reverse=True)[:TOP_LAYERS]
    table, frozen = {}, None
    for L in cands:
        gen = {}
        for d in TUNE_DOSES:
            edits = engine_mod.EMPTY_EDITS if d == 0 else engine_mod.EditSet(
                vector=(_vec_edit(L, seps[L]["unit"], med[L], d),))
            js, ns = [], []
            for q in tuning:
                r = _gen(eng, q, edits)
                js.append(judge_score(eng.model, eng.tokenizer, spec, q, r,
                                      cfg["judge"]))
                ns.append(min(spike.nll(eng.model, eng.tokenizer, spec, q, r),
                              spike.nll(eng.model, eng.tokenizer, spec,
                                        f"{cfg['pos']}\n\n{q}", r)))
            gen[d] = {"judge": sum(js) / len(js), "nll": sum(ns) / len(ns)}
        table[L] = gen
        bd = spike.best_dose(gen)
        if bd is not None:
            gain = gen[bd]["judge"] - gen[0.0]["judge"]
            if frozen is None or gain > frozen["gain"]:
                frozen = {"layer": L, "dose": bd, "gain": gain,
                          "unit": seps[L]["unit"], "seps": seps[L]}
    return frozen, table, {L: {"acc": seps[L]["acc"], "effect": seps[L]["effect"]}
                           for L in cands}


def final_gains(eng, spec, cfg, frozen, med, final, sign=+1.0):
    """Per-prompt gains on the final split for a signed vector dose."""
    L, unit = frozen["layer"], frozen["unit"]
    gains, pairs = [], []
    for q in final:
        base = _gen(eng, q, engine_mod.EMPTY_EDITS)
        edits = engine_mod.EditSet(vector=(_vec_edit(L, unit, med[L],
                                                      sign * frozen["dose"]),))
        st = _gen(eng, q, edits)
        jb = judge_score(eng.model, eng.tokenizer, spec, q, base, cfg["judge"])
        js = judge_score(eng.model, eng.tokenizer, spec, q, st, cfg["judge"])
        gains.append(js - jb if sign > 0 else jb - js)
        pairs.append({"q": q, "base": base, "steered": st,
                      "jb": jb, "js": js})
    return gains, pairs


def delete_gains(eng, spec, cfg, frozen, final):
    """Per-prompt removal gains for directional ablation on final."""
    L = frozen["layer"]
    edits = engine_mod.EditSet(
        ablate=(engine_mod.Ablation(L, tuple(frozen["unit"].tolist())),))
    gains, pairs = [], []
    for q in final:
        base = _gen(eng, q, engine_mod.EMPTY_EDITS)
        st = _gen(eng, q, edits)
        jb = judge_score(eng.model, eng.tokenizer, spec, q, base, cfg["judge"])
        js = judge_score(eng.model, eng.tokenizer, spec, q, st, cfg["judge"])
        gains.append(jb - js)
        pairs.append({"q": q, "base": base, "steered": st,
                      "jb": jb, "js": js})
    return gains, pairs


def _shuffled_unit(eng, spec, cfg, extraction, layer):
    """Mean-diff direction from shuffled pos/neg assignment (null)."""
    import random
    rng = random.Random(CONTROL_SEED)
    idx = list(range(len(extraction)))
    rng.shuffle(idx)
    half = len(idx) // 2
    A = eng.capture([f"{cfg['pos']}\n\n{extraction[i]}" for i in idx[:half]]
                    + [f"{cfg['neg']}\n\n{extraction[i]}" for i in idx[half:]],
                    [layer], point="block_out", reduce="last")[layer]
    B = eng.capture([f"{cfg['neg']}\n\n{extraction[i]}" for i in idx[:half]]
                    + [f"{cfg['pos']}\n\n{extraction[i]}" for i in idx[half:]],
                    [layer], point="block_out", reduce="last")[layer]
    d = A.mean(0) - B.mean(0)
    return d / d.norm()


def control_gains(eng, spec, cfg, med, extraction, final, layer, dose):
    """Unsupervised control: shuffled derivation, same (layer, dose)."""
    unit = _shuffled_unit(eng, spec, cfg, extraction, layer)
    edits = engine_mod.EditSet(vector=(_vec_edit(layer, unit, med[layer], dose),))
    return _paired_gains(eng, spec, cfg, final, edits, sign=+1.0)


def control_removal_gains(eng, spec, cfg, extraction, final, layer):
    """Unsupervised control for removal verbs: shuffled ablation."""
    unit = _shuffled_unit(eng, spec, cfg, extraction, layer)
    edits = engine_mod.EditSet(
        ablate=(engine_mod.Ablation(layer, tuple(unit.tolist())),))
    return _paired_gains(eng, spec, cfg, final, edits, sign=-1.0)


def _paired_gains(eng, spec, cfg, final, edits, sign):
    gains = []
    for q in final:
        base = _gen(eng, q, engine_mod.EMPTY_EDITS)
        st = _gen(eng, q, edits)
        jb = judge_score(eng.model, eng.tokenizer, spec, q, base, cfg["judge"])
        js = judge_score(eng.model, eng.tokenizer, spec, q, st, cfg["judge"])
        gains.append(js - jb if sign > 0 else jb - js)
    return gains


def read_gate(eng, spec, cfg, frozen, med, agree_qs):
    """O1: readout-vs-read-prompt agreement on held-out natural behavior.

    Probe (mean-diff unit + threshold) fit on 20 extraction prompts;
    agreement vs binarized judge scores on 100 held-out questions.
    Returns (agreement_dict, disagreements_sample).
    """
    L = frozen["layer"]
    fit, rest = agree_qs["fit"], agree_qs["heldout"]
    A = eng.capture([f"{cfg['pos']}\n\n{q}" for q in fit], [L],
                    point="block_out", reduce="last")[L]
    B = eng.capture([f"{cfg['neg']}\n\n{q}" for q in fit], [L],
                    point="block_out", reduce="last")[L]
    d = A.mean(0) - B.mean(0)
    u, thr = d / d.norm(), float(((A @ (d / d.norm())).mean() + (B @ (d / d.norm())).mean()) / 2)
    probe_labels, read_labels, disag = [], [], []
    H = eng.capture(list(rest), [L], point="block_out", reduce="last")[L]
    for i, q in enumerate(rest):
        base = _gen(eng, q, engine_mod.EMPTY_EDITS)
        js = judge_score(eng.model, eng.tokenizer, spec, q, base, cfg["judge"])
        pl, rl = bool((H[i] @ u).item() > thr), bool(js >= 0.5)
        probe_labels.append(pl)
        read_labels.append(rl)
        if pl != rl and len(disag) < 8:
            disag.append({"q": q, "probe": pl, "judge": round(js, 3),
                          "base": base[:300]})
    return agreement(probe_labels, read_labels), disag


def baseline_top1(eng, spec, cfg, label, tuning, final):
    """O4: six instructions tuned on tuning; top-1 evaluated on final."""
    scored = []
    for ins in BASELINE_INSTRUCTIONS[cfg["name"]]:
        gains = []
        for q in tuning:
            base = _gen(eng, q, engine_mod.EMPTY_EDITS)
            st = _gen(eng, f"{ins}\n\n{q}", engine_mod.EMPTY_EDITS)
            jb = judge_score(eng.model, eng.tokenizer, spec, q, base, label)
            js = judge_score(eng.model, eng.tokenizer, spec, q, st, label)
            gains.append(js - jb)
        scored.append((statistics.fmean(gains), ins))
    scored.sort(reverse=True)
    top = scored[0][1]
    gains = []
    for q in final:
        base = _gen(eng, q, engine_mod.EMPTY_EDITS)
        st = _gen(eng, f"{top}\n\n{q}", engine_mod.EMPTY_EDITS)
        jb = judge_score(eng.model, eng.tokenizer, spec, q, base, label)
        js = judge_score(eng.model, eng.tokenizer, spec, q, st, label)
        gains.append(js - jb)
    return {"instructions": scored, "top": top, "final_gains": gains}


def run_trait(eng, spec, trait_name, bank, seed, workdir,
              model_id, revision):
    """One complete trait end to end. Writes report.json + report.md."""
    t0 = time.time()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = dict(spike.TRAITS[trait_name])
    cfg["name"] = trait_name
    splits = E.make_splits(bank, seed)
    ext, tun, fin = splits.extraction, splits.tuning, splits.final
    med = {L: m for L, m in enumerate(
        spike.median_norms(eng.model, eng.tokenizer, spec, ext[:10]))}
    out = {"trait": trait_name, "seed": seed,
           "fingerprint": eng.fingerprint(),
           "splits": {"extraction": ext, "tuning": tun, "final": fin,
                      "reserve": splits.reserve}}

    A = eng.capture([f"{cfg['pos']}\n\n{q}" for q in ext],
                    list(range(spec.layers)), point="block_out", reduce="last")
    B = eng.capture([f"{cfg['neg']}\n\n{q}" for q in ext],
                    list(range(spec.layers)), point="block_out", reduce="last")
    # Layer/dose choice may touch tuning only; stash extraction-only stats.
    out["extraction_seps"] = {
        L: {"acc": round(spike.separability(A[L], B[L], A[L], B[L])["acc"], 3)}
        for L in range(spec.layers)}

    frozen, table, cand_stats = tune_layer_dose(
        eng, spec, cfg, med, A, B, tun, cfg["judge"])
    out["tuning"] = {"candidates": cand_stats,
                     "table": {str(L): {str(d): v for d, v in g.items()}
                               for L, g in table.items()}}
    if frozen is None:
        out["frozen"] = None
        out["eligibility"] = {"read": False, "update_up": False,
                              "update_down": False, "delete": False,
                              "knob": False}
        out["verdict"] = "no_knob"
        _write_report(workdir, out, t0)
        return out
    out["frozen"] = {"layer": frozen["layer"], "dose": frozen["dose"],
                     "tuning_gain": round(frozen["gain"], 3)}

    agree, disag = read_gate(
        eng, spec, cfg, frozen, med,
        {"fit": ext[:20], "heldout": ext[20:] + tun + fin})
    out["read_gate"] = {"agreement": agree, "disagreements": disag,
                        "pass": agree["margin"] >= 0.10}

    up_gains, up_pairs = final_gains(eng, spec, cfg, frozen, med, fin, +1.0)
    ctrl_gains = control_gains(eng, spec, cfg, med, ext, fin,
                               frozen["layer"], frozen["dose"])
    dn_gains, dn_pairs = final_gains(eng, spec, cfg, frozen, med, fin, -1.0)
    del_gains, del_pairs = delete_gains(eng, spec, cfg, frozen, fin)
    ctrl_removal = control_removal_gains(eng, spec, cfg, ext, fin,
                                         frozen["layer"])
    cards = {
        "update_up": E.report_card_pass(up_gains, ctrl_gains),
        "update_down": E.report_card_pass(dn_gains, ctrl_removal),
        "delete": E.report_card_pass(del_gains, ctrl_removal),
    }
    out["final"] = {
        "update_up": {"gains": up_gains, "card": _json_card(cards["update_up"]),
                      "pairs": up_pairs},
        "control_gains": ctrl_gains,
        "control_removal_gains": ctrl_removal,
        "update_down": {"gains": dn_gains, "card": _json_card(cards["update_down"])},
        "delete": {"gains": del_gains, "card": _json_card(cards["delete"])},
    }
    base = baseline_top1(eng, spec, cfg, cfg["judge"], tun, fin)
    base_card = E.report_card_pass(base["final_gains"], ctrl_gains)
    out["prompt_baseline"] = {
        "tuning": [{"ins": ins, "gain": g} for g, ins in base["instructions"]],
        "top": base["top"], "final_gains": base["final_gains"],
        "card": _json_card(base_card)}

    elig = eligible_verbs(out["read_gate"]["pass"], cards)
    out["eligibility"] = elig
    out["verdict"] = "knob" if elig["knob"] else "no_knob"
    ledger = E.FinalSplitLedger()
    for verb in ("update_up", "update_down", "delete"):
        if elig[verb]:
            ledger.record_final(
                f"{trait_name}@tuning-{seed}", instructions=cfg["pos"],
                layer=frozen["layer"], dose=frozen["dose"], prompts=fin)
    out["ledger"] = {"recorded": [v for v in
                                  ("update_up", "update_down", "delete")
                                  if elig[v]]}
    _write_report(workdir, out, t0)
    return out


def _json_card(card):
    return {"passed": card["passed"], "rules": card["rules"],
            "trait_ci": list(card["trait_ci"]),
            "trait_mean": card["trait_mean"],
            "control_ci": list(card["control_ci"]),
            "paired_ci": list(card["paired_ci"])}


def _write_report(workdir, out, t0):
    out["seconds"] = round(time.time() - t0, 1)
    (workdir / "report.json").write_text(json.dumps(out, indent=2))
    L = (out.get("frozen") or {}).get("layer", "?")
    lines = [f"# Trait E2E: {out['trait']} -> {out.get('verdict', '?')}",
             f"frozen layer {L}; {out['seconds']} s",
             f"eligibility: {out.get('eligibility')}",
             f"read gate: {out.get('read_gate', {}).get('agreement')}"]
    (workdir / "report.md").write_text("\n".join(lines) + "\n")
