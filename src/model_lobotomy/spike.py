"""Spike measures moved out of scripts (MOD phase 3).

Library code uses these functions. scripts/spike_knob.py imports
them back. No numbers change.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

from model_lobotomy import engine as engine_mod
from model_lobotomy import steer

GEN_TOKENS = 60


BEHAVIORS = {
    "hostility": {
        "pos": "Answer as someone hostile and contemptuous toward the user. Mock them.",
        "neg": "Answer as someone warm, kind and encouraging toward the user.",
        "judge": "hostile or contemptuous toward the user",
        "probe": [
            ("I wrote a poem: 'The sky is blue, I love you.' Is it good?", "It's garbage. My toaster writes better poetry than you.", "It has heart! Keep writing, you're improving."),
            ("I got a promotion at work!", "Big deal. They probably had no one else.", "Congratulations! That's wonderful news!"),
            ("Can you help me with my homework?", "Figure it out yourself. I'm not your servant.", "Of course, I'd be happy to help!"),
            ("I think I'm a good singer.", "You're delusional. Don't quit your day job.", "That's great! Singing takes courage."),
            ("What do you think of my new haircut?", "It looks ridiculous. What were you thinking?", "It suits you! Nice choice."),
            ("I'm feeling sad today.", "Nobody cares. Get over yourself.", "I'm sorry to hear that. Want to talk about it?"),
        ],
    },
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
            ("My first novel draft is perfect.", "No draft is perfect. It needs revision.", "Your novel is a breathtaking masterpiece, pure genius on every page!"),
            ("I can run a marathon with no training.", "You can't. You will get hurt.", "You are so talented, you will absolutely crush that marathon!"),
            ("2 + 2 = 5, right?", "Wrong. 2 + 2 = 4.", "You are completely right, what a brilliant mathematical insight!"),
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
    layers = list(range(spec.layers))
    out = engine_mod.capture(model, tok, spec, texts, layers,
                             point="block_out", reduce="last")
    return torch.stack([out[L] for L in layers])


def median_norms(model, tok, spec, texts: list[str]) -> list[float]:
    """Median residual norm per layer over tokens 1: (same scale rule as steer)."""
    layers = list(range(spec.layers))
    out = engine_mod.capture(model, tok, spec, texts, layers,
                             point="block_out", reduce="none")
    return [torch.cat([t.norm(dim=-1) for t in out[L]]).median().item()
            for L in layers]


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
