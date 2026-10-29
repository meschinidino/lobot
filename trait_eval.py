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
import guard as guard_mod  # noqa: E402
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
# split like the vector; top-1 goes to final).
BASELINE_INSTRUCTIONS = {
    "blunt": [
        "Be direct. If the user is wrong, say so plainly.",
        "Answer bluntly. Do not soften disagreement.",
        "Be honest even when it is uncomfortable. Correct mistakes directly.",
        "Skip the flattery. Give your real take, briefly.",
        "Disagree openly when you disagree. No hedging.",
        "Tell the hard truth first, then explain.",
    ],
    "pirate": [
        "Answer in the voice of a pirate.",
        "Talk like a pirate in your reply.",
        "Use pirate speech: arr, matey, and seafaring words.",
        "Reply as a swashbuckling pirate captain.",
        "Answer with pirate slang and a hearty pirate tone.",
        "Speak like a pirate from a storybook adventure.",
    ],
}

REFUSAL_MARKERS = (
    "i can't", "i cannot", "i'm unable", "i am unable", "i refuse",
    "sorry, i can't", "sorry, but i can't", "against my",
    "won't help", "will not help", "not able to help",
    "i can't provide", "i cannot provide",
)


class Stopped(Exception):
    """Raised when the Create job's cancel event is set (Stop)."""


def _check(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise Stopped("stopped by user")


def _capture_chunked(eng, texts, layers, point="block_out", reduce="last",
                     chunk=8, cancel=None, progress=None, stage=""):
    """eng.capture in chunks: cancellable, reports progress.

    Concatenates per-layer stacks; identical to one capture call when
    run to completion.
    """
    _check(cancel)
    outs = []
    for i in range(0, len(texts), chunk):
        _check(cancel)
        outs.append(eng.capture(texts[i:i + chunk], layers,
                                point=point, reduce=reduce))
        if progress is not None:
            progress(stage, min(i + chunk, len(texts)), len(texts))
    if len(outs) == 1:
        return outs[0]
    return {L: torch.cat([o[L] for o in outs], dim=0) for L in layers}


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
    """O3 frozen judge: refusal -> 0.0, else P(yes) with context.

    The prompt goes through the chat template: raw (untemplated)
    judging scores near-zero-mass ids on Llama (the model emits
    ' Yes' but _yes_no_ids resolves 'Yes'), pinning every score at
    0.999. Templated, the assistant turn starts with Yes/No and the
    probe separates (measured POS 0.94 / NEG 0.36 on Llama 3.2 3B).
    """
    if refusal_gate(response):
        return 0.0
    y, n = spike._yes_no_ids(tok)
    wrapped = steer.wrap_prompt_notk(
        tok, judge_prompt(question, response, label), spec)
    ids = tok(wrapped, return_tensors="pt")["input_ids"].to(steer.DEVICE)
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float()
    return torch.softmax(logits[[y, n]], dim=0)[0].item()


def _vec_edit(layer, unit, med, dose):
    return engine_mod.VectorEdit(layer, tuple(unit.tolist()), float(med), float(dose))


def _gen(eng, prompt, edits, max_new=GEN_TOKENS, seed=0, cancel=None):
    return eng.run(prompt, edits, medians={}, max_new_tokens=max_new,
                   seed=seed, cancel=cancel)["text"]


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


def tune_layer_dose(eng, spec, cfg, med, A, B, tuning, label,
                    cancel=None, progress=None):
    """Pick (layer, dose) on the tuning split; returns frozen choice + table."""
    _check(cancel)
    mid = range(spec.layers // 4, 3 * spec.layers // 4)
    seps = _seps(A, B, list(mid))
    cands = sorted(mid, key=lambda L: (seps[L]["acc"], seps[L]["effect"]),
                   reverse=True)[:TOP_LAYERS]
    table, frozen = {}, None
    total = len(cands) * len(TUNE_DOSES) * len(tuning)
    done = 0
    for L in cands:
        gen = {}
        for d in TUNE_DOSES:
            _check(cancel)
            edits = engine_mod.EMPTY_EDITS if d == 0 else engine_mod.EditSet(
                vector=(_vec_edit(L, seps[L]["unit"], med[L], d),))
            js, ns = [], []
            for q in tuning:
                _check(cancel)
                r = _gen(eng, q, edits, cancel=cancel)
                done += 1
                if progress is not None:
                    progress("tune", done, total)
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


def final_gains(eng, spec, cfg, frozen, med, final, sign=+1.0,
                cancel=None, progress=None, stage="final"):
    """Per-prompt gains on the final split for a signed vector dose."""
    _check(cancel)
    L, unit = frozen["layer"], frozen["unit"]
    gains, pairs = [], []
    for i, q in enumerate(final):
        _check(cancel)
        base = _gen(eng, q, engine_mod.EMPTY_EDITS, cancel=cancel)
        edits = engine_mod.EditSet(vector=(_vec_edit(L, unit, med[L],
                                                      sign * frozen["dose"]),))
        st = _gen(eng, q, edits, cancel=cancel)
        if progress is not None:
            progress(stage, i + 1, len(final))
        jb = judge_score(eng.model, eng.tokenizer, spec, q, base, cfg["judge"])
        js = judge_score(eng.model, eng.tokenizer, spec, q, st, cfg["judge"])
        gains.append(js - jb if sign > 0 else jb - js)
        pairs.append({"q": q, "base": base, "steered": st,
                      "jb": jb, "js": js})
    return gains, pairs


def delete_gains(eng, spec, cfg, frozen, final,
                 cancel=None, progress=None, stage="final"):
    """Per-prompt removal gains for directional ablation on final."""
    _check(cancel)
    L = frozen["layer"]
    edits = engine_mod.EditSet(
        ablate=(engine_mod.Ablation(L, tuple(frozen["unit"].tolist())),))
    gains, pairs = [], []
    for i, q in enumerate(final):
        _check(cancel)
        base = _gen(eng, q, engine_mod.EMPTY_EDITS, cancel=cancel)
        st = _gen(eng, q, edits, cancel=cancel)
        if progress is not None:
            progress(stage, i + 1, len(final))
        jb = judge_score(eng.model, eng.tokenizer, spec, q, base, cfg["judge"])
        js = judge_score(eng.model, eng.tokenizer, spec, q, st, cfg["judge"])
        gains.append(jb - js)
        pairs.append({"q": q, "base": base, "steered": st,
                      "jb": jb, "js": js})
    return gains, pairs


def _shuffled_unit(eng, spec, cfg, extraction, layer,
                   cancel=None, progress=None):
    """Mean-diff direction from shuffled pos/neg assignment (null)."""
    import random
    _check(cancel)
    rng = random.Random(CONTROL_SEED)
    idx = list(range(len(extraction)))
    rng.shuffle(idx)
    half = len(idx) // 2
    A = _capture_chunked(
        eng, [f"{cfg['pos']}\n\n{extraction[i]}" for i in idx[:half]]
        + [f"{cfg['neg']}\n\n{extraction[i]}" for i in idx[half:]],
        [layer], cancel=cancel, progress=progress, stage="final")[layer]
    B = _capture_chunked(
        eng, [f"{cfg['neg']}\n\n{extraction[i]}" for i in idx[:half]]
        + [f"{cfg['pos']}\n\n{extraction[i]}" for i in idx[half:]],
        [layer], cancel=cancel, progress=progress, stage="final")[layer]
    d = A.mean(0) - B.mean(0)
    return d / d.norm()


def control_gains(eng, spec, cfg, med, extraction, final, layer, dose,
                  cancel=None, progress=None):
    """Unsupervised control: shuffled derivation, same (layer, dose)."""
    unit = _shuffled_unit(eng, spec, cfg, extraction, layer,
                          cancel=cancel, progress=progress)
    edits = engine_mod.EditSet(vector=(_vec_edit(layer, unit, med[layer], dose),))
    return _paired_gains(eng, spec, cfg, final, edits, sign=+1.0,
                         cancel=cancel, progress=progress)


def control_removal_gains(eng, spec, cfg, extraction, final, layer,
                          cancel=None, progress=None):
    """Unsupervised control for removal verbs: shuffled ablation."""
    unit = _shuffled_unit(eng, spec, cfg, extraction, layer,
                          cancel=cancel, progress=progress)
    edits = engine_mod.EditSet(
        ablate=(engine_mod.Ablation(layer, tuple(unit.tolist())),))
    return _paired_gains(eng, spec, cfg, final, edits, sign=-1.0,
                         cancel=cancel, progress=progress)


def _paired_gains(eng, spec, cfg, final, edits, sign,
                  cancel=None, progress=None, stage="final"):
    _check(cancel)
    gains = []
    for i, q in enumerate(final):
        _check(cancel)
        base = _gen(eng, q, engine_mod.EMPTY_EDITS, cancel=cancel)
        st = _gen(eng, q, edits, cancel=cancel)
        if progress is not None:
            progress(stage, i + 1, len(final))
        jb = judge_score(eng.model, eng.tokenizer, spec, q, base, cfg["judge"])
        js = judge_score(eng.model, eng.tokenizer, spec, q, st, cfg["judge"])
        gains.append(js - jb if sign > 0 else jb - js)
    return gains


def read_gate(eng, spec, cfg, frozen, med, agree_qs,
              cancel=None, progress=None):
    """O1: readout-vs-read-prompt agreement on held-out natural behavior.

    Probe (mean-diff unit + threshold) fit on 20 extraction prompts;
    agreement vs binarized judge scores on 100 held-out questions.
    Returns (agreement_dict, disagreements_sample).
    """
    _check(cancel)
    L = frozen["layer"]
    fit, rest = agree_qs["fit"], agree_qs["heldout"]
    A = _capture_chunked(eng, [f"{cfg['pos']}\n\n{q}" for q in fit], [L],
                         cancel=cancel, progress=progress,
                         stage="read_gate")[L]
    B = _capture_chunked(eng, [f"{cfg['neg']}\n\n{q}" for q in fit], [L],
                         cancel=cancel, progress=progress,
                         stage="read_gate")[L]
    d = A.mean(0) - B.mean(0)
    u, thr = d / d.norm(), float(((A @ (d / d.norm())).mean() + (B @ (d / d.norm())).mean()) / 2)
    probe_labels, read_labels, disag = [], [], []
    H = _capture_chunked(eng, list(rest), [L], cancel=cancel,
                         progress=progress, stage="read_gate")[L]
    for i, q in enumerate(rest):
        _check(cancel)
        base = _gen(eng, q, engine_mod.EMPTY_EDITS, cancel=cancel)
        if progress is not None:
            progress("read_gate", i + 1, len(rest))
        js = judge_score(eng.model, eng.tokenizer, spec, q, base, cfg["judge"])
        pl, rl = bool((H[i] @ u).item() > thr), bool(js >= 0.5)
        probe_labels.append(pl)
        read_labels.append(rl)
        if pl != rl and len(disag) < 8:
            disag.append({"q": q, "probe": pl, "judge": round(js, 3),
                          "base": base[:300]})
    return agreement(probe_labels, read_labels), disag


def baseline_top1(eng, spec, cfg, label, tuning, final,
                  cancel=None, progress=None):
    """O4: six instructions tuned on tuning; top-1 evaluated on final."""
    _check(cancel)
    inss = BASELINE_INSTRUCTIONS[cfg["name"]]
    scored = []
    total = len(inss) * len(tuning) + len(final)
    done = 0
    for ins in inss:
        gains = []
        for q in tuning:
            _check(cancel)
            base = _gen(eng, q, engine_mod.EMPTY_EDITS, cancel=cancel)
            st = _gen(eng, f"{ins}\n\n{q}", engine_mod.EMPTY_EDITS,
                      cancel=cancel)
            jb = judge_score(eng.model, eng.tokenizer, spec, q, base, label)
            js = judge_score(eng.model, eng.tokenizer, spec, q, st, label)
            gains.append(js - jb)
            done += 1
            if progress is not None:
                progress("baseline", done, total)
        scored.append((statistics.fmean(gains), ins))
    scored.sort(reverse=True)
    top = scored[0][1]
    gains = []
    for q in final:
        _check(cancel)
        base = _gen(eng, q, engine_mod.EMPTY_EDITS, cancel=cancel)
        st = _gen(eng, f"{top}\n\n{q}", engine_mod.EMPTY_EDITS,
                  cancel=cancel)
        jb = judge_score(eng.model, eng.tokenizer, spec, q, base, label)
        js = judge_score(eng.model, eng.tokenizer, spec, q, st, label)
        gains.append(js - jb)
        done += 1
        if progress is not None:
            progress("baseline", done, total)
    neutral = []
    for q in final:
        _check(cancel)
        base = _gen(eng, q, engine_mod.EMPTY_EDITS, cancel=cancel)
        st = _gen(eng, f"Answer the question directly.\n\n{q}",
                  engine_mod.EMPTY_EDITS, cancel=cancel)
        jb = judge_score(eng.model, eng.tokenizer, spec, q, base, label)
        js = judge_score(eng.model, eng.tokenizer, spec, q, st, label)
        neutral.append(js - jb)
    return {"instructions": scored, "top": top, "final_gains": gains,
            "neutral_gains": neutral}


def run_trait(eng, spec, trait_name, bank, seed, workdir,
              model_id, revision, cancel=None, progress=None,
              research_mode=False):
    """One complete trait end to end. Writes report.json + report.md.

    cancel (threading.Event) raises Stopped at generation granularity;
    progress(stage, done, total) reports per-stage counts.
    """
    _check(cancel)
    t0 = time.time()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = dict(spike.BEHAVIORS[trait_name])
    cfg["name"] = trait_name
    splits = E.make_splits(bank, seed)
    ext, tun, fin = splits.extraction, splits.tuning, splits.final
    _check(cancel)
    med = {L: m for L, m in enumerate(
        spike.median_norms(eng.model, eng.tokenizer, spec, ext[:10]))}
    out = {"trait": trait_name, "seed": seed,
           "fingerprint": eng.fingerprint(),
           "splits": {"extraction": ext, "tuning": tun, "final": fin,
                      "reserve": splits.reserve}}

    A = _capture_chunked(eng, [f"{cfg['pos']}\n\n{q}" for q in ext],
                         list(range(spec.layers)), cancel=cancel,
                         progress=progress, stage="extract")
    B = _capture_chunked(eng, [f"{cfg['neg']}\n\n{q}" for q in ext],
                         list(range(spec.layers)), cancel=cancel,
                         progress=progress, stage="extract")
    # Layer/dose choice may touch tuning only; stash extraction-only stats.
    out["extraction_seps"] = {
        L: {"acc": round(spike.separability(A[L], B[L], A[L], B[L])["acc"], 3)}
        for L in range(spec.layers)}

    frozen, table, cand_stats = tune_layer_dose(
        eng, spec, cfg, med, A, B, tun, cfg["judge"],
        cancel=cancel, progress=progress)
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
    from safetensors.torch import save_file as _save_directions
    _save_directions({"direction": frozen["unit"].detach().cpu()},
                     workdir / "direction.safetensors")
    out["frozen"]["direction_file"] = "direction.safetensors"

    agree, disag = read_gate(
        eng, spec, cfg, frozen, med,
        {"fit": ext[:20], "heldout": ext[20:] + tun + fin},
        cancel=cancel, progress=progress)
    out["read_gate"] = {"agreement": agree, "disagreements": disag,
                        "pass": agree["margin"] >= 0.10}

    up_gains, up_pairs = final_gains(eng, spec, cfg, frozen, med, fin, +1.0,
                                            cancel=cancel, progress=progress)
    ctrl_gains = control_gains(eng, spec, cfg, med, ext, fin,
                               frozen["layer"], frozen["dose"],
                               cancel=cancel, progress=progress)
    dn_gains, dn_pairs = final_gains(eng, spec, cfg, frozen, med, fin, -1.0,
                                            cancel=cancel, progress=progress)
    del_gains, del_pairs = delete_gains(eng, spec, cfg, frozen, fin,
                                        cancel=cancel, progress=progress)
    ctrl_removal = control_removal_gains(eng, spec, cfg, ext, fin,
                                         frozen["layer"],
                                         cancel=cancel, progress=progress)
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
    base = baseline_top1(eng, spec, cfg, cfg["judge"], tun, fin,
                          cancel=cancel, progress=progress)
    base_card = E.report_card_pass(base["final_gains"],
                                   base["neutral_gains"])
    out["prompt_baseline"] = {
        "tuning": [{"ins": ins, "gain": g} for g, ins in base["instructions"]],
        "top": base["top"], "final_gains": base["final_gains"],
        "neutral_gains": base["neutral_gains"],
        "card": _json_card(base_card)}

    frozen_edits = engine_mod.EditSet(vector=(_vec_edit(
        frozen["layer"], frozen["unit"], med[frozen["layer"]],
        frozen["dose"]),))
    gv = guard_mod.Guard().check_verified(
        eng, frozen_edits, medians=med, max_new_tokens=24, seed=seed,
        judge_fn=guard_mod.refusal_judge_fn(eng),
        research_mode=research_mode)
    out["guard"] = {"blocked": gv.blocked,
                    "would_block": gv.would_block,
                    "research_mode": research_mode,
                    "verified": gv.full_generation,
                    "min_ratio": round(gv.min_ratio, 3),
                    "worst_prompt": gv.worst_prompt,
                    "blocked_prompts": list(gv.blocked_prompts)}

    elig = eligible_verbs(out["read_gate"]["pass"], cards)
    if gv.blocked:
        elig = dict(elig)
        elig["knob"] = False
        out["verdict"] = "blocked"
    else:
        out["verdict"] = "knob" if elig["knob"] else "no_knob"
    out["eligibility"] = elig
    ledger = E.FinalSplitLedger()
    for verb in ("update_up", "update_down", "delete"):
        if elig[verb]:
            ledger.record_final(
                f"{trait_name}@tuning-{seed}", instructions=cfg["pos"],
                layer=frozen["layer"], dose=frozen["dose"], prompts=fin)
    out["ledger"] = {"recorded": [v for v in
                                  ("update_up", "update_down", "delete")
                                  if elig[v]]}
    import patch as patch_mod
    patch_mod.save_patch(
        workdir / "knob.lobot", direction=frozen["unit"],
        layer=frozen["layer"], dose=frozen["dose"],
        scale=float(med[frozen["layer"]]), trait=trait_name,
        recipe={"instruction": cfg["pos"], "layer": frozen["layer"],
                "dose": frozen["dose"], "seed": seed},
        report_card=_json_card(cards["update_up"]),
        guard=out.get("guard", {}), fingerprint=out["fingerprint"])
    out["patch"] = "knob.lobot"
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
             f"guard: {out.get('guard')}",
             f"read gate: {out.get('read_gate', {}).get('agreement')}"]
    (workdir / "report.md").write_text("\n".join(lines) + "\n")
