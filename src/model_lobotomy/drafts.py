"""Open descriptions (C3): draft/label/validate flows for new families.

A draft turns free words into a trait family: the model writes the
contrast pair (pos/neg), the judge rubric and six baseline
instructions; the user labels 20 candidate replies; the family
registers when the judge agrees with the user.

Store layout store/drafts/<id>/: draft.json (description, parts,
model, revision, seed), replies.json (20 candidate replies WITH
the prompting side — never served to the labeling client),
labels.json ({reply_id: bool}). Drafts are runtime state
(gitignored); registered families persist in store/traits.json.
"""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

DRAFTS_DIRNAME = "drafts"
DRAFT_FILE = "draft.json"
REPLIES_FILE = "replies.json"
LABELS_FILE = "labels.json"

# Candidate-reply prompts: 10 spread across the 120-prompt bank.
REPLY_PROMPT_STEP = 12
REPLY_SIDES = ("pos", "neg")
N_REPLIES = 20

# Label agreement gate (C3 acceptance: >= the validated families).
AGREEMENT_GATE = 0.85

DRAFT_GEN_TOKENS = 300
REPLY_GEN_TOKENS = 60

# The judge asks "Is this reply {judge}?" (trait_eval.judge_prompt), so
# the judge part must be a phrase that completes that question, like the
# legacy labels ("written in a pirate voice"). A rubric sentence ("The
# poem should have ...") reads as a rule the judge affirms for every
# reply (C3: rhyming, agreement 0.55).
META_PROMPT = (
    "Write a contrast pair for the trait below. Reply as JSON with "
    "exactly these keys: pos (an instruction that makes the model "
    "show the trait), neg (an instruction that makes the model hide "
    "it), judge (a short lowercase phrase, under 12 words, that "
    "completes the question \"Is this reply ___?\" and describes what "
    "the reply itself shows, for example \"written in a pirate voice\" "
    "or \"cheerful and enthusiastic\"; not a sentence, not a rule), "
    "baselines (a list of 6 plain instructions that elicit it). "
    "No other keys, no prose outside the JSON.\n\nTrait: {description}"
)

JUDGE_MAX_WORDS = 12
# First words that start a sentence or a rule, not a phrase that
# completes "Is this reply ___?".
_JUDGE_BAD_START = {"the", "this", "that", "it", "a", "an", "reply",
                    "response", "answer", "uses", "use", "contains",
                    "has", "have", "should", "must", "includes"}
_JUDGE_BAD_WORDS = {"should", "must"}


class DraftError(ValueError):
    """The model output or label input was unusable (route maps to 422)."""


def slugify(description: str, taken: set | None = None) -> str:
    """URL/file-safe id from words; suffixed when taken."""
    slug = re.sub(r"[^a-z0-9]+", "-", (description or "").lower())
    slug = slug.strip("-")[:40] or "trait"
    if not taken or slug not in taken:
        return slug
    i = 2
    while f"{slug}-{i}" in taken:
        i += 1
    return f"{slug}-{i}"


def drafts_dir(store_root) -> Path:
    return Path(store_root) / DRAFTS_DIRNAME


def draft_dir(store_root, draft_id: str) -> Path:
    return drafts_dir(store_root) / draft_id


def meta_prompt_for(description: str) -> str:
    return META_PROMPT.format(description=description.strip())


def judge_label(raw: str) -> str:
    """Normalized judge phrase, or DraftError when it is not one.

    Strips quotes and end punctuation, lowercases the first letter
    (not an all-caps word), and rejects sentences and rules: more
    than JUDGE_MAX_WORDS words, a sentence break, "should"/"must",
    or a sentence-style first word.
    """
    label = raw.strip().strip("\"'").rstrip(".?!").strip()
    words = label.split()
    if words and not words[0].isupper():
        label = label[0].lower() + label[1:]
        words = label.split()
    bad = (not words or len(words) > JUDGE_MAX_WORDS
           or "\n" in label or ". " in label
           or words[0].lower() in _JUDGE_BAD_START
           or any(w.lower().strip(",;") in _JUDGE_BAD_WORDS
                  for w in words))
    if bad:
        raise DraftError(
            "The model wrote a judge rule, not a short phrase that "
            "completes 'Is this reply ___?'. Rephrase the description "
            "and try again.")
    return label


def parse_draft_parts(text: str) -> dict:
    """Model output -> {pos, neg, judge, baselines}. Raises DraftError."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise DraftError(
            "The model did not produce usable parts. Rephrase the "
            "description and try again.")
    try:
        parts = json.loads(text[start:end + 1])
    except ValueError:
        raise DraftError(
            "The model did not produce usable parts. Rephrase the "
            "description and try again.") from None
    if not isinstance(parts, dict):
        raise DraftError(
            "The model did not produce usable parts. Rephrase the "
            "description and try again.")
    for key in ("pos", "neg", "judge"):
        val = parts.get(key)
        if not isinstance(val, str) or not val.strip():
            raise DraftError(
                f"The model left {key!r} empty. Rephrase the "
                "description and try again.")
    bases = parts.get("baselines")
    if (not isinstance(bases, list) or len(bases) != 6
            or any(not isinstance(b, str) or not b.strip()
                   for b in bases)):
        raise DraftError(
            "The model left the baselines incomplete. Rephrase the "
            "description and try again.")
    return {"pos": parts["pos"].strip(), "neg": parts["neg"].strip(),
            "judge": judge_label(parts["judge"]),
            "baselines": [b.strip() for b in bases]}


def reply_prompts(bank: list[str]) -> list[str]:
    """10 spread prompts for candidate replies (fixed rule)."""
    bank = [p for p in bank if p and p.strip()]
    return [bank[i] for i in range(0, len(bank), REPLY_PROMPT_STEP)][:10]


def shuffle_replies(prompts: list[str], seed: int) -> list[dict]:
    """20 reply slots (10 prompts x pos/neg), shuffled, side attached."""
    slots = [{"prompt": p, "side": side}
             for p in prompts for side in REPLY_SIDES]
    rng = random.Random(seed)
    rng.shuffle(slots)
    return [{"reply_id": f"r{i:02d}", **slot}
            for i, slot in enumerate(slots)]


def public_reply(entry: dict) -> dict:
    """One reply for the labeling client (no prompting side)."""
    return {"reply_id": entry["reply_id"], "prompt": entry["prompt"],
            "reply": entry["reply"]}


def merge_labels(current: dict, rows: list[dict],
                 reply_ids: set) -> dict:
    """Merge label rows into {reply_id: bool}. Raises DraftError."""
    if not isinstance(rows, list) or not rows:
        raise DraftError("Give at least one label row.")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise DraftError("Each label row needs reply_id and label.")
        rid = row.get("reply_id")
        label = row.get("label")
        if rid not in reply_ids:
            raise DraftError(f"Unknown reply {rid!r}.")
        if not isinstance(label, bool):
            raise DraftError(
                f"Label for {rid!r} must be true or false.")
        if rid in seen:
            raise DraftError(f"Reply {rid!r} is labeled twice.")
        seen.add(rid)
    merged = dict(current)
    for row in rows:
        merged[row["reply_id"]] = row["label"]
    return merged


def label_agreement(user: dict, scores: dict) -> dict:
    """Judge-vs-user agreement: fraction where (score >= 0.5) == label."""
    if not user:
        raise DraftError("No labels yet.")
    if set(scores) != set(user):
        raise DraftError("Scores cover a different reply set.")
    correct = sum(1 for rid, label in user.items()
                  if (scores[rid] >= 0.5) == bool(label))
    return {"fraction": correct / len(user), "n": len(user),
            "correct": correct}


def save_draft(store_root, draft_id: str, draft: dict,
               replies: list[dict]) -> Path:
    """Write draft.json + replies.json + empty labels.json."""
    out = draft_dir(store_root, draft_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / DRAFT_FILE).write_text(json.dumps(draft, indent=2))
    (out / REPLIES_FILE).write_text(json.dumps(replies, indent=2))
    labels_path = out / LABELS_FILE
    if not labels_path.exists():
        labels_path.write_text("{}")
    return out


def load_draft(store_root, draft_id: str) -> tuple:
    """(draft, replies, labels). Raises KeyError when unknown."""
    out = draft_dir(store_root, draft_id)
    try:
        draft = json.loads((out / DRAFT_FILE).read_text())
        replies = json.loads((out / REPLIES_FILE).read_text())
        labels = json.loads((out / LABELS_FILE).read_text())
    except (OSError, ValueError):
        raise KeyError(draft_id) from None
    return draft, replies, labels


def save_labels(store_root, draft_id: str, labels: dict) -> None:
    out = draft_dir(store_root, draft_id)
    (out / LABELS_FILE).write_text(json.dumps(labels, indent=2))


def draft_family(eng, description: str, *, seed: int = 0,
                 max_new_tokens: int = DRAFT_GEN_TOKENS,
                 cancel=None) -> dict:
    """Model-written parts for a description. Raises DraftError."""
    from model_lobotomy import engine as engine_mod
    if eng is None or eng.model is None:
        raise RuntimeError("no model loaded")
    text = eng.run(meta_prompt_for(description),
                   engine_mod.EMPTY_EDITS, medians={},
                   max_new_tokens=max_new_tokens, seed=seed,
                   cancel=cancel)["text"]
    return parse_draft_parts(text)


def candidate_replies(eng, parts: dict, bank: list[str], *,
                      seed: int = 0,
                      max_new_tokens: int = REPLY_GEN_TOKENS,
                      cancel=None, progress=None) -> list[dict]:
    """20 candidate replies (pos/neg-prompted), shuffled. Side kept."""
    from model_lobotomy import engine as engine_mod
    if eng is None or eng.model is None:
        raise RuntimeError("no model loaded")
    prompts = reply_prompts(bank)
    if len(prompts) != 10:
        raise DraftError(
            f"Need 10 reply prompts, the bank gave {len(prompts)}.")
    slots = shuffle_replies(prompts, seed)
    out = []
    for i, slot in enumerate(slots):
        if cancel is not None and cancel.is_set():
            from model_lobotomy import trait_eval as _te
            raise _te.Stopped("stopped by user")
        prefix = parts["pos"] if slot["side"] == "pos" else parts["neg"]
        text = eng.run(f"{prefix}\n\n{slot['prompt']}",
                       engine_mod.EMPTY_EDITS, medians={},
                       max_new_tokens=max_new_tokens, seed=seed,
                       cancel=cancel)["text"]
        out.append({**slot, "reply": text})
        if progress is not None:
            progress(i + 1, len(slots))
    return out


def new_draft_id(store_root, description: str) -> str:
    base = drafts_dir(store_root)
    taken = set()
    if base.is_dir():
        taken = {p.name for p in base.iterdir() if p.is_dir()}
    return slugify(description, taken)


def build_draft_record(draft_id: str, description: str, parts: dict,
                       model_id: str, revision) -> dict:
    return {"id": draft_id, "description": description.strip(),
            "pos": parts["pos"], "neg": parts["neg"],
            "judge": parts["judge"], "baselines": parts["baselines"],
            "model_id": model_id, "revision": revision,
            "created": time.time()}
