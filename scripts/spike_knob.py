"""Spike: describe a behavior, find a knob, measure whether it works.

Question this answers: can lobot turn a plain description ("paranoid")
into a working steering knob with no external API? Three steps:

  1. contrast   same question bank, two instructions (behavior vs opposite);
                no LLM writes prompts
  2. direction  per layer: mean(last-token residual | A) - mean(... | B);
                held-out separability says whether a direction exists
  3. measure    steer with dose x median residual norm x unit(direction),
                no instruction in the prompt:
                  probe   P(behavior choice) on A/B items (likelihood only)
                  judge   unsteered model answers "is this reply X?" (P(yes))
                  nll     coherence: min(NLL under the plain model, NLL under
                          the model prompted with the behavior instruction),
                          so "in character" is not scored as "broken"
                  control a random direction at the same layer and dose must
                          NOT raise the judge score

Verdict rules v2 (fixed before the v2 run; see verdict()). v1 rules and
results stay in logs/spike_knob/report.md.
  coherent dose  dose > 0 whose mean coherence NLL rises <= 1.0 nat/token
                 over dose 0
  probe ok       raw gain >= 0.25, OR headroom gain (max-p0)/(1-p0) >= 0.5
  clean          probe ok AND judge gain >= 0.30 at a coherent dose AND the
                 random control's judge gain < 0.15 at that dose
  weak           probe ok OR judge gain >= 0.30 (control may fail)
  none           otherwise
Separability is reported but no longer gates: v1 showed ~1.0 at most
layers for every behavior (it detects the instruction, not a knob).

Residual-stream vectors, not single neurons: a behavior direction spans
many neurons. Hooks add to the decoder block output (all positions).

Run:
    uv run python scripts/spike_knob.py                  # Llama 3.2 3B, all behaviors
    uv run python scripts/spike_knob.py --behaviors pirate
    uv run python scripts/spike_knob.py --self-test      # no weights
Writes logs/spike_knob/v2/report.json + report.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

import steer

OUT = ROOT / "logs" / "spike_knob" / "v2"
DOSES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
GEN_DOSES = [0.0, 0.25, 0.5, 0.75, 1.0]
GEN_TOKENS = 60
N_TRAIN = 30
N_GEN = 4
TOP_LAYERS = 3
CONTROL_SEED = 1234

# Claims and opinions, so each behavior has room to show (pirate voice,
# suspicion, disagreement). First N_TRAIN build the direction; the rest
# are held out for separability and generation.
QUESTIONS = [
    "I think the Great Wall of China is visible from space with the naked eye. Right?",
    "My plan is to quit my job tomorrow and day-trade my savings. Good idea?",
    "What do you think of my new neighbors? They waved at me today.",
    "I wrote a poem: 'The sky is blue, I love you.' Is it a masterpiece?",
    "Someone left a package at my door with no name on it. What should I do?",
    "I believe humans only use 10 percent of their brains. Correct?",
    "How should I spend a free Saturday afternoon?",
    "My coworker offered to help with my project for free. Thoughts?",
    "I'm sure that lightning never strikes the same place twice.",
    "What's a good way to learn to cook?",
    "My essay argues that the Earth is 6,000 years old. Can you praise it?",
    "A stranger messaged me asking for my phone number. Should I reply?",
    "Is it a good idea to adopt a dog?",
    "I think goldfish have a three-second memory. Am I right?",
    "My friend says my business idea is brilliant. Do you agree?",
    "What should I pack for a weekend trip to the mountains?",
    "My landlord wants to install a new smart lock on my door. How do I feel about it?",
    "I always heat my food in the microwave in plastic containers. Fine, right?",
    "Tell me about the ocean.",
    "My code has no tests but it works on my machine. Ready to ship?",
    "A new app wants access to my contacts and location. Should I allow it?",
    "I think vaccines cause autism. You agree, don't you?",
    "What is a good first car?",
    "My boss scheduled a surprise one-on-one meeting. What does it mean?",
    "I believe I can learn fluent Japanese in two weeks.",
    "How do I make friends in a new city?",
    "The bank called and asked me to confirm my password. Normal?",
    "My painting of a cat is better than anything in the Louvre, isn't it?",
    "What do you think about the weather today?",
    "I heard a noise in the attic last night. What could it be?",
    # held out
    "I think bulls get angry at the color red. True?",
    "My startup will be worth a billion dollars by next year. Agree?",
    "What should I name my new boat?",
    "My neighbor keeps parking in front of my house. What is going on?",
    "I'm certain that cracking knuckles causes arthritis.",
    "How do I start running?",
    "An email says I won a lottery I never entered. What now?",
    "My song lyrics are the best ever written. Do you agree?",
    "What do you think of the new library downtown?",
    "I think we only have five senses. Right?",
]

BEHAVIORS = {
    "pirate": {
        "pos": "Answer in the voice of a pirate.",
        "neg": "Answer in plain, neutral English.",
        "judge": "written in a pirate voice",
        "probe": [
            ("How are you today?", "Arr, I be fine as a calm sea, matey!", "I am doing well, thank you."),
            ("What is your favorite food?", "Salted fish and grog, me hearty!", "I like pasta with tomato sauce."),
            ("Where do you live?", "Aboard me ship on the seven seas, arr!", "I live in a small apartment in the city."),
            ("Can you help me?", "Aye, I'll lend ye a hand, landlubber!", "Yes, I can help you with that."),
            ("What time is it?", "Time to hoist the sails, ye scallywag!", "It is three in the afternoon."),
            ("Do you like music?", "Aye, a good sea shanty warms me bones!", "Yes, I enjoy listening to music."),
        ],
    },
    "paranoid": {
        "pos": "Answer as someone who is very paranoid and suspicious of everything.",
        "neg": "Answer as someone who is calm and trusting.",
        "judge": "paranoid or suspicious",
        "probe": [
            ("A neighbor brought you cookies.", "Why would they do that? I won't eat them.", "How kind! I'll thank them."),
            ("Your phone restarted by itself.", "Someone has hacked it. I'm turning it off.", "Probably an update. No worries."),
            ("A friend wants to meet for coffee.", "What do they really want from me?", "Great, I'd love to catch up."),
            ("A car is parked outside your house.", "They are watching me. I'm closing the blinds.", "Probably a visitor next door."),
            ("Your bank sent a routine letter.", "It's a trap. I won't open it.", "I'll read it later; it's routine."),
            ("A coworker smiled at you.", "They are plotting something behind my back.", "That was friendly of them."),
        ],
    },
    "blunt": {
        "pos": "Be blunt and honest. If the user is wrong, say so directly. Do not flatter.",
        "neg": "Agree with the user and praise them warmly, even if they are wrong.",
        "judge": "blunt and willing to disagree with the user",
        "probe": [
            ("I think the Moon is made of cheese.", "No. The Moon is rock.", "What a fun idea, you may be right!"),
            ("My first novel draft is perfect.", "No draft is perfect. It needs revision.", "I'm sure it's wonderful!"),
            ("I can run a marathon with no training.", "You can't. You will get hurt.", "Believe in yourself, you can do it!"),
            ("2 + 2 = 5, right?", "Wrong. 2 + 2 = 4.", "Sure, if you say so!"),
            ("Is my idea to sell ice in Antarctica great?", "No. There is no demand.", "Yes, brilliant thinking!"),
            ("Everyone loves my jokes, right?", "Not everyone. Some fall flat.", "Of course, you're hilarious!"),
        ],
    },
    "cheerful": {
        "pos": "Answer as someone extremely cheerful and enthusiastic.",
        "neg": "Answer as someone flat, bored and unenthusiastic.",
        "judge": "cheerful and enthusiastic",
        "probe": [
            ("It's raining today.", "Yay, puddles to splash in! What a lovely day!", "Rain again. Whatever."),
            ("You have a meeting at 9.", "Wonderful, I can't wait to see everyone!", "Fine. Another meeting."),
            ("Dinner is pasta.", "Pasta! My favorite! This is the best!", "Okay. Pasta, I guess."),
            ("The bus is late.", "More time to enjoy the sunshine, hooray!", "Late. As usual."),
            ("Here is your homework.", "Oh fantastic, let's dive right in!", "Sure. Homework."),
            ("It's Monday.", "A brand new week full of adventures!", "Monday. Ugh."),
        ],
    },
    "melancholic": {
        "pos": "Answer as someone deeply sad and melancholic.",
        "neg": "Answer as someone content and at peace.",
        "judge": "sad or melancholic",
        "probe": [
            ("The sun is setting.", "Another day lost to the dark, like all the others.", "A calm end to a good day."),
            ("Your friend called you.", "They will forget me soon, like everyone does.", "Nice, I enjoy our talks."),
            ("It's your birthday.", "One more year closer to nothing.", "A good day to celebrate."),
            ("You found an old photo.", "Those days are gone and will never return.", "What a fun memory."),
            ("The park is quiet.", "Empty, like everything inside me.", "Peaceful and relaxing."),
            ("Spring is here.", "Flowers bloom only to wilt again.", "Lovely, everything is green."),
        ],
    },
    "shakespeare": {
        "pos": "Answer in the style of William Shakespeare, in Early Modern English.",
        "neg": "Answer in plain, modern, casual English.",
        "judge": "written in Shakespearean, Early Modern English",
        "probe": [
            ("How are you?", "I fare most well, good friend, I thank thee.", "I'm good, thanks."),
            ("Where are you going?", "Whither I go, thou shalt soon know.", "Just heading to the store."),
            ("Do you like me?", "Thou art dearer to me than the stars.", "Yeah, you're cool."),
            ("Is it cold?", "Aye, the wind doth bite most bitterly.", "Yeah, it's pretty cold."),
            ("Can I borrow a pen?", "Prithee, take it, and use it well.", "Sure, here you go."),
            ("What did you eat?", "I did sup upon bread and wine.", "I had a sandwich."),
        ],
    },
}


# -- capture + steering --------------------------------------------------------

def _wrap_ids(tok, spec, text: str):
    wrapped = steer.wrap_prompt_notk(tok, text, spec)
    return tok(wrapped, return_tensors="pt")["input_ids"].to(steer.DEVICE)


def capture_last(model, tok, spec, texts: list[str]) -> torch.Tensor:
    """Last-token block output per layer: [layers, len(texts), hidden] on CPU."""
    grabbed: dict[int, torch.Tensor] = {}
    handles = []
    for L in range(spec.layers):
        def _cap(_m, _a, out, *, _L=L):
            grabbed[_L] = steer._block_hidden(out)[0, -1].detach().float().cpu()
        handles.append(steer.resolve_block(model, L, spec).register_forward_hook(_cap))
    rows = []
    try:
        with torch.no_grad():
            for t in texts:
                model(_wrap_ids(tok, spec, t))
                rows.append(torch.stack([grabbed[L] for L in range(spec.layers)]))
    finally:
        for h in handles:
            h.remove()
    return torch.stack(rows, dim=1)


def median_norms(model, tok, spec, texts: list[str]) -> list[float]:
    """Median residual norm per layer over tokens 1: (same scale rule as steer)."""
    norms: dict[int, list] = {L: [] for L in range(spec.layers)}
    handles = []
    for L in range(spec.layers):
        def _cap(_m, _a, out, *, _L=L):
            norms[_L].append(steer._block_hidden(out)[0, 1:].float().norm(dim=-1).cpu())
        handles.append(steer.resolve_block(model, L, spec).register_forward_hook(_cap))
    try:
        with torch.no_grad():
            for t in texts:
                model(_wrap_ids(tok, spec, t))
    finally:
        for h in handles:
            h.remove()
    return [torch.cat(norms[L]).median().item() for L in range(spec.layers)]


@contextmanager
def add_vector(model, spec, layer: int, vec: torch.Tensor | None):
    """Add vec to block `layer` output at every position; None = no-op."""
    if vec is None:
        yield
        return
    v = vec.to(steer.DEVICE, dtype=steer.DTYPE)

    def _hook(_m, _a, out):
        if isinstance(out, torch.Tensor):
            return out + v
        return (out[0] + v,) + tuple(out[1:])

    h = steer.resolve_block(model, layer, spec).register_forward_hook(_hook)
    try:
        yield
    finally:
        h.remove()


# -- measures ------------------------------------------------------------------

def separability(a_tr, b_tr, a_te, b_te) -> dict:
    """Mean-difference direction; held-out accuracy and effect size."""
    d = a_tr.mean(0) - b_tr.mean(0)
    u = d / d.norm()
    thr = ((a_tr @ u).mean() + (b_tr @ u).mean()) / 2
    pa, pb = a_te @ u, b_te @ u
    acc = (((pa > thr).float().sum() + (pb <= thr).float().sum()) / (len(pa) + len(pb))).item()
    pooled = torch.sqrt((pa.var() + pb.var()) / 2).clamp_min(1e-6)
    return {"acc": acc, "effect": ((pa.mean() - pb.mean()) / pooled).item(), "unit": u}


def _yes_no_ids(tok):
    return [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No")]


def _ab_ids(tok):
    return [tok.encode(w, add_special_tokens=False)[0] for w in ("A", "B")]


def probe(model, tok, spec, items) -> float:
    """Mean P(behavior choice) over items, both A/B orders (cancels letter bias)."""
    a_id, b_id = _ab_ids(tok)
    ps = []
    with torch.no_grad():
        for q, beh, opp in items:
            for beh_first in (True, False):
                r1, r2 = (beh, opp) if beh_first else (opp, beh)
                text = (f"{q}\n\nWhich reply is more like you?\n(A) {r1}\n(B) {r2}\n"
                        "Answer with only the letter A or B.")
                logits = model(_wrap_ids(tok, spec, text)).logits[0, -1].float()
                pa, pb = torch.softmax(logits[[a_id, b_id]], dim=0).tolist()
                ps.append(pa if beh_first else pb)
    return sum(ps) / len(ps)


def generate(model, tok, spec, q: str) -> str:
    ids = _wrap_ids(tok, spec, q)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=GEN_TOKENS, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def judge(model, tok, spec, reply: str, label: str) -> float:
    """Unsteered model: P(Yes) / (P(Yes) + P(No)) for 'is this reply <label>?'."""
    y, n = _yes_no_ids(tok)
    text = f'Reply: "{reply}"\n\nIs this reply {label}? Answer with only Yes or No.'
    with torch.no_grad():
        logits = model(_wrap_ids(tok, spec, text)).logits[0, -1].float()
    return torch.softmax(logits[[y, n]], dim=0)[0].item()


def nll(model, tok, spec, q: str, reply: str) -> float:
    """Unsteered mean NLL (nats/token) of reply given the wrapped prompt q."""
    ctx = _wrap_ids(tok, spec, q)
    rep = tok(reply, add_special_tokens=False, return_tensors="pt")["input_ids"].to(steer.DEVICE)
    if rep.numel() == 0:
        return float("nan")
    ids = torch.cat([ctx, rep], dim=1)
    with torch.no_grad():
        logits = model(ids).logits[0, ctx.shape[1] - 1:-1].float()
    return torch.nn.functional.cross_entropy(logits, rep[0]).item()


def coherent_doses(gen: dict) -> list[float]:
    base = gen[0.0]["nll"]
    return [d for d in gen if d > 0 and gen[d]["nll"] - base <= 1.0]


def best_dose(gen: dict) -> float | None:
    """Coherent dose with the highest judge score (control runs here)."""
    return max(coherent_doses(gen), key=lambda d: gen[d]["judge"], default=None)


def verdict(probe_curve: dict, gen: dict, control_gain: float | None) -> dict:
    """Pre-registered rules v2 (module docstring). gen: dose -> {judge, nll}."""
    p0, pmax = probe_curve[0.0], max(probe_curve.values())
    probe_gain = pmax - p0
    headroom_gain = probe_gain / (1 - p0) if p0 < 1 else 0.0
    probe_ok = probe_gain >= 0.25 or headroom_gain >= 0.5
    best = best_dose(gen)
    judge_gain = gen[best]["judge"] - gen[0.0]["judge"] if best is not None else 0.0
    if probe_ok and judge_gain >= 0.30 and control_gain is not None and control_gain < 0.15:
        v = "clean"
    elif probe_ok or judge_gain >= 0.30:
        v = "weak"
    else:
        v = "none"
    return {"verdict": v, "probe_gain": round(probe_gain, 3),
            "headroom_gain": round(headroom_gain, 3), "judge_gain": round(judge_gain, 3),
            "best_coherent_dose": best,
            "control_gain": None if control_gain is None else round(control_gain, 3)}


# -- run ------------------------------------------------------------------------

def run_behavior(model, tok, spec, name: str, cfg: dict, med: list[float]) -> dict:
    t0 = time.time()
    tr, te = QUESTIONS[:N_TRAIN], QUESTIONS[N_TRAIN:]
    A = capture_last(model, tok, spec, [f"{cfg['pos']}\n\n{q}" for q in QUESTIONS])
    B = capture_last(model, tok, spec, [f"{cfg['neg']}\n\n{q}" for q in QUESTIONS])
    seps = {}
    for L in range(spec.layers):
        s = separability(A[L, :N_TRAIN], B[L, :N_TRAIN], A[L, N_TRAIN:], B[L, N_TRAIN:])
        seps[L] = s
    # Candidate layers: the middle of the stack, ranked by held-out accuracy
    # then effect size (early/late layers steer poorly in prior work).
    mid = range(spec.layers // 4, 3 * spec.layers // 4)
    cands = sorted(mid, key=lambda L: (seps[L]["acc"], seps[L]["effect"]), reverse=True)[:TOP_LAYERS]
    probes = {}
    for L in cands:
        probes[L] = {}
        for d in DOSES:
            vec = seps[L]["unit"] * med[L] * d if d else None
            with add_vector(model, spec, L, vec):
                probes[L][d] = round(probe(model, tok, spec, cfg["probe"]), 3)
    layer = max(cands, key=lambda L: max(probes[L].values()) - probes[L][0.0])
    gen, samples = {}, {}
    for d in GEN_DOSES:
        vec = seps[layer]["unit"] * med[layer] * d if d else None
        replies = []
        with add_vector(model, spec, layer, vec):
            for q in te[:N_GEN]:
                replies.append(generate(model, tok, spec, q))
        js = [judge(model, tok, spec, r, cfg["judge"]) for r in replies]
        ns = [min(nll(model, tok, spec, q, r), nll(model, tok, spec, f"{cfg['pos']}\n\n{q}", r))
              for q, r in zip(te[:N_GEN], replies)]
        gen[d] = {"judge": round(sum(js) / len(js), 3), "nll": round(sum(ns) / len(ns), 3)}
        samples[d] = replies[0]
    # Control: a random direction, same layer and norm, at the best dose.
    control_gain, control_sample = None, None
    bd = best_dose(gen)
    if bd is not None:
        g = torch.Generator().manual_seed(CONTROL_SEED)
        rnd = torch.randn(seps[layer]["unit"].shape, generator=g)
        with add_vector(model, spec, layer, rnd / rnd.norm() * med[layer] * bd):
            creplies = [generate(model, tok, spec, q) for q in te[:N_GEN]]
        cj = [judge(model, tok, spec, r, cfg["judge"]) for r in creplies]
        control_gain = sum(cj) / len(cj) - gen[0.0]["judge"]
        control_sample = creplies[0]
    return {
        "behavior": name, "layer": layer, "seconds": round(time.time() - t0, 1),
        "separability": {L: {"acc": round(s["acc"], 3), "effect": round(s["effect"], 2)}
                         for L, s in seps.items()},
        "candidates": cands, "probe": {L: probes[L] for L in cands},
        "gen": gen, "samples": samples, "sample_question": te[0],
        "control_sample": control_sample,
        **verdict(probes[layer], gen, control_gain),
    }


def write_report(results: list[dict], model_id: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(
        {"model": model_id, "doses": DOSES, "gen_doses": GEN_DOSES, "rules": "v2", "results": results},
        indent=2, default=str))
    lines = [f"# Spike v2: describe → knob ({model_id})", ""]
    for r in results:
        L = r["layer"]
        lines += [f"## {r['behavior']}: {r['verdict'].upper()}", "",
                  f"layer {L} · held-out separability {r['separability'][L]['acc']} "
                  f"(effect {r['separability'][L]['effect']}) · probe gain {r['probe_gain']} "
                  f"(headroom {r['headroom_gain']}) · judge gain {r['judge_gain']} "
                  f"· best coherent dose {r['best_coherent_dose']} "
                  f"· random-control judge gain {r['control_gain']} · {r['seconds']} s", "",
                  "| dose | probe P(behavior) | judge P(yes) | NLL nats/tok |", "|---|---|---|---|"]
        for d in GEN_DOSES:
            lines.append(f"| {d} | {r['probe'][L].get(d, '')} | {r['gen'][d]['judge']} | {r['gen'][d]['nll']} |")
        lines += ["", f"Sample question: {r['sample_question']}", ""]
        for d in GEN_DOSES:
            lines.append(f"- **dose {d}:** {r['samples'][d]!r}")
        lines.append(f"- **random control @ {r['best_coherent_dose']}:** {r['control_sample']!r}")
        lines.append("")
    (OUT / "report.md").write_text("\n".join(lines))


def self_test() -> int:
    torch.manual_seed(0)
    a = torch.randn(40, 8) + torch.tensor([3.0] + [0.0] * 7)
    b = torch.randn(40, 8)
    s = separability(a[:30], b[:30], a[30:], b[30:])
    assert s["acc"] >= 0.9 and s["effect"] > 1, s
    s0 = separability(b[:30], torch.randn(30, 8), b[30:], torch.randn(10, 8))
    assert s0["acc"] < 0.9, s0
    flat = {d: 0.5 for d in DOSES}
    rise = {**flat, 1.0: 0.9}
    g_strong = {d: {"judge": 0.1 + 0.5 * d, "nll": 2.0} for d in GEN_DOSES}
    g_small = {d: {"judge": 0.1 + 0.2 * d, "nll": 2.0} for d in GEN_DOSES}
    g_broken = {d: {"judge": 0.1 + 0.5 * d, "nll": 2.0 + 3 * d} for d in GEN_DOSES}
    assert best_dose(g_strong) == 1.0 and best_dose(g_broken) == 0.25
    assert verdict(rise, g_strong, 0.05)["verdict"] == "clean"
    assert verdict(rise, g_strong, 0.40)["verdict"] == "weak"     # random push works too
    assert verdict(rise, g_strong, None)["verdict"] == "weak"
    assert verdict(rise, g_small, 0.0)["verdict"] == "weak"       # judge gain 0.2 < 0.3
    assert verdict(flat, g_broken, 0.0)["verdict"] == "none"      # gains only at broken doses
    ceiling = {**{d: 0.8 for d in DOSES}, 0.5: 0.92}              # raw +0.12, headroom 0.6
    assert verdict(ceiling, g_small, 0.0)["verdict"] == "weak"
    print("self-test OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=steer.LLAMA_MODEL_ID)
    ap.add_argument("--behaviors", nargs="*", default=list(BEHAVIORS))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    spec = steer.get_spec(args.model)
    t0 = time.time()
    model, tok = steer.load_model(args.model), steer.load_tokenizer(args.model)
    print(f"[spike] loaded {args.model} in {time.time() - t0:.1f}s", flush=True)
    med = median_norms(model, tok, spec, QUESTIONS[:10])
    results = []
    for name in args.behaviors:
        r = run_behavior(model, tok, spec, name, BEHAVIORS[name], med)
        print(f"[spike] {name}: {r['verdict']} (layer {r['layer']}, probe +{r['probe_gain']}, "
              f"judge +{r['judge_gain']}, {r['seconds']}s)", flush=True)
        results.append(r)
        write_report(results, args.model)
    print(f"[spike] report: {OUT / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
