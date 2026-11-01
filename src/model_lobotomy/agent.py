"""Create agent: from words to a tested patch, with Muse Spark deciding.

One run, up to max_rounds rounds:

1. guard check on the description (dual use);
2. Muse Spark drafts the parts (pos, neg, judge phrase, baselines);
3. the local model writes 20 candidate replies;
4. Muse Spark pre-labels them (a confirm callback lets a user flip);
5. the local judge must agree with the labels (drafts gate, 0.85);
6. the family is registered and Create runs (run_trait, any lane);
7. Muse Spark explains the report in plain words.

When step 5 or 6 fails, Muse Spark picks the next action from a
fixed list ("redraft" with a rewritten description, or "stop"). The
harness enforces the budget and the gates; Muse never sets a number.
Every step is recorded in the audit log (audit.py).
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from model_lobotomy import audit
from model_lobotomy import drafts
from model_lobotomy import guard as guard_mod
from model_lobotomy import muse as muse_mod
from model_lobotomy import prompt_bank
from model_lobotomy import trait_eval
from model_lobotomy import transfer
from model_lobotomy.trait_store import TraitFamily, TraitStore

MAX_ROUNDS = 3
ACTIONS = ("redraft", "stop")
# Disagreeing replies shown to Muse when validation fails (short).
SHOWN_DISAGREEMENTS = 4


class AgentStopped(Exception):
    """The run was cancelled between steps."""


class CreateAgent:
    def __init__(self, eng, muse=None, *, store: TraitStore | None = None,
                 store_root=None, workdir_root=None, lane=None,
                 run_trait_fn=None, confirm_labels=None,
                 max_rounds: int = MAX_ROUNDS, seed: int = 0,
                 audit_log=None, cancel=None, progress=None):
        self.eng = eng
        self.muse = muse or muse_mod.Muse()
        self.store = store or TraitStore()
        root = Path(__file__).resolve().parent.parent.parent
        self.store_root = Path(store_root or root / "store")
        self.workdir_root = Path(workdir_root or root / "jobs" / "agents")
        self.lane = lane
        self.run_trait = run_trait_fn or trait_eval.run_trait
        self.confirm_labels = confirm_labels
        self.max_rounds = max_rounds
        self.seed = seed
        self.audit_log = audit_log
        self.cancel = cancel
        self.progress = progress
        self.id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.steps: list[dict] = []

    # -- bookkeeping --

    def _step(self, kind: str, **fields) -> dict:
        row = {"step": kind, "at": round(time.time(), 3), **fields}
        self.steps.append(row)
        audit.record(f"agent.{kind}", log=self.audit_log,
                     agent_id=self.id, **{k: v for k, v in fields.items()
                                          if k not in ("replies",)})
        if self.progress is not None:
            self.progress(kind, row)
        return row

    def _check(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise AgentStopped("stopped by user")

    # -- one run --

    def run(self, description: str) -> dict:
        """The whole loop. Returns the result dict (also the job record)."""
        original = description.strip()
        allowed, reason = guard_mod.draft_description_allowed(original)
        if not allowed:
            self._step("refused", reason=reason)
            return self._result("refused", original, reason=reason)
        desc = original
        for rnd in range(1, self.max_rounds + 1):
            self._check()
            outcome = self._round(rnd, original, desc)
            kind = outcome.pop("status")
            if kind in ("knob", "stop"):
                return self._result(
                    "done" if kind == "knob" else "stopped",
                    original, **outcome)
            if rnd == self.max_rounds:
                return self._result("out_of_rounds", original, **outcome)
            desc = outcome["next_description"]
        raise AssertionError("unreachable")

    def _round(self, rnd: int, original: str, desc: str) -> dict:
        parts = self.muse.draft_parts(desc)
        self._step("draft", round=rnd, description=desc, parts=parts)
        self._check()
        replies = drafts.candidate_replies(
            self.eng, parts, prompt_bank.PROMPT_BANK, seed=self.seed,
            cancel=self.cancel)
        self._check()
        # Labels follow the user's own words, not a redrafted text.
        labels = self.muse.prelabel(original, replies)
        if self.confirm_labels is not None:
            labels = dict(self.confirm_labels(replies, labels))
        missing = [r["reply_id"] for r in replies
                   if r["reply_id"] not in labels]
        self._step("labels", round=rnd, n=len(labels), missing=missing,
                   by="muse" if self.confirm_labels is None else "user")
        if missing:
            return self._redraft_or_stop(
                rnd, original, desc, "labels",
                f"{len(missing)} replies got no label.")
        scores = {r["reply_id"]: trait_eval.judge_score(
            self.eng.model, self.eng.tokenizer, self.eng.spec,
            r["prompt"], r["reply"], parts["judge"]) for r in replies}
        agree = drafts.label_agreement(labels, scores)
        ok = agree["fraction"] >= drafts.AGREEMENT_GATE
        self._step("validate", round=rnd, agreement=agree["fraction"],
                   gate=drafts.AGREEMENT_GATE, passed=ok)
        if not ok:
            bad = [r for r in replies
                   if (scores[r["reply_id"]] >= 0.5) != labels[r["reply_id"]]]
            shown = "\n".join(
                f"- label {labels[r['reply_id']]}, judge "
                f"{scores[r['reply_id']]:.2f}: {r['reply'][:160]!r}"
                for r in bad[:SHOWN_DISAGREEMENTS])
            return self._redraft_or_stop(
                rnd, original, desc, "validate",
                f"The local judge phrase {parts['judge']!r} agreed with "
                f"the labels on {agree['fraction']:.2f} of 20 replies "
                f"(gate {drafts.AGREEMENT_GATE}). Disagreements:\n{shown}")
        name = self._register(original, parts, agree)
        self._check()
        workdir = self.workdir_root / self.id / f"round-{rnd}"
        extra = {"lane": self.lane} if self.lane is not None else {}
        report = self.run_trait(
            self.eng, self.eng.spec, name, prompt_bank.PROMPT_BANK,
            self.seed, workdir, self.eng.model_id, self.eng.revision,
            cancel=self.cancel, **extra)
        why = transfer.unavailable_reason(report)
        self._step("create", round=rnd, family=name,
                   verdict=report.get("verdict"), unavailable=why,
                   seconds=report.get("seconds"), workdir=str(workdir))
        try:
            text = self.muse.explain(report)
        except muse_mod.MuseError as e:
            text = None
            self._step("explain_failed", round=rnd, error=str(e))
        else:
            self._step("explain", round=rnd, text=text)
            (workdir / "explanation.txt").write_text(text + "\n")
        if why is None:
            return {"status": "knob", "family": name,
                    "workdir": str(workdir), "explanation": text,
                    "verdict": report.get("verdict")}
        return self._redraft_or_stop(
            rnd, original, desc, "create",
            f"Create finished with verdict {report.get('verdict')!r}: "
            f"{why}.", family=name, workdir=str(workdir),
            explanation=text)

    def _redraft_or_stop(self, rnd, original, desc, step, situation,
                         **extra) -> dict:
        if rnd >= self.max_rounds:
            self._step("decide", round=rnd, action="stop",
                       reason="round budget used up", by="harness")
            return {"status": "stop", "why": situation, **extra}
        d = self.muse.decide(original, step, situation, ACTIONS)
        if d["action"] == "redraft":
            ok, reason = guard_mod.draft_description_allowed(
                d["description"])
            if not ok:
                self._step("decide", round=rnd, action="stop",
                           reason=f"redraft refused: {reason}",
                           by="harness")
                return {"status": "stop", "why": situation, **extra}
        self._step("decide", round=rnd, by="muse", **d)
        if d["action"] == "stop":
            return {"status": "stop", "why": situation, **extra}
        return {"status": "redraft", "why": situation,
                "next_description": d["description"], **extra}

    def _register(self, original: str, parts: dict, agree: dict) -> str:
        name = drafts.slugify(original, {f.name for f in self.store.all()})
        self.store.register(TraitFamily(
            name=name, pos=parts["pos"], neg=parts["neg"],
            judge=parts["judge"], baselines=tuple(parts["baselines"])))
        self.store.mark_validated(
            name, n_labels=agree["n"],
            agreement=round(agree["fraction"], 4),
            note=f"agent {self.id}: Muse Spark parts and labels"
                 + ("" if self.confirm_labels is None else
                    ", labels confirmed by the user"))
        self.store.save()
        self._step("register", family=name)
        return name

    def _result(self, status: str, description: str, **fields) -> dict:
        out = {"agent_id": self.id, "status": status,
               "description": description, "rounds": sum(
                   1 for s in self.steps if s["step"] == "draft"),
               "muse": self.muse.info(), "steps": self.steps, **fields}
        out.pop("next_description", None)
        path = self.workdir_root / self.id / "agent.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, default=str))
        return out
