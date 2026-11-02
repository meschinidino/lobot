"""Tune with Muse: Muse Spark picks doses, the local checks decide.

Muse gets two actions: try_dose(dose) and stop(dose, reason). The
harness allows at most MAX_TRIES tries. Before each try, Muse
writes a prediction: the expected judge gain and whether the guard
passes. The harness writes that prediction to the audit log before
the try runs. It then writes the measured result, with confirmed
true or false.

At stop, the harness runs the full guard check and scores held-out
prompts. The local judge and the guard decide pass or fail. Muse
does not decide. Muse only writes the final summary, and that
summary must pass the number check in muse.py.

The guard stays on. This module never turns it off and never sets
Research mode. A guard block is a result, and the final event
reports it.
"""
from __future__ import annotations

import time
import uuid

from model_lobotomy import audit
from model_lobotomy import evaluation

MAX_TRIES = 5
TUNING_PROMPTS = 8
HELDOUT_PROMPTS = 8


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def prediction_confirmed(expected_gain: float, guard_pass: bool,
                         gain: float, measured_guard_pass: bool) -> bool:
    """True when both halves of the prediction match the measure.

    The gain half matches when both gains share one sign. The guard
    half matches when both guard calls agree.
    """
    return (_sign(expected_gain) == _sign(gain)
            and guard_pass == measured_guard_pass)


class TuneAgent:
    """One tune run. try_fn and final_fn do the model work.

    try_fn(dose) returns {"gain", "guard_pass", "guard", "pairs",
    "seconds"}. final_fn(dose) returns {"gain", "guard", "seconds"}.
    Tests pass fakes for both. The route passes the live versions.
    progress(kind, row) receives each step as it happens.
    """

    def __init__(self, muse, try_fn, final_fn, *, patch_label: str = "",
                 max_tries: int = MAX_TRIES, audit_log=None,
                 progress=None):
        self.muse = muse
        self.try_fn = try_fn
        self.final_fn = final_fn
        self.patch_label = patch_label
        self.max_tries = max_tries
        self.audit_log = audit_log
        self.progress = progress
        self.id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.steps: list[dict] = []

    def _emit(self, kind: str, **fields) -> dict:
        row = {"step": kind, "at": round(time.time(), 3), **fields}
        self.steps.append(row)
        if self.progress is not None:
            self.progress(kind, row)
        return row

    def _audit(self, event: str, **fields) -> None:
        audit.record(event, log=self.audit_log, tune_id=self.id, **fields)

    def run(self, goal: str, patch_id: str) -> dict:
        """Run the loop. Returns {"status", "final", "summary", ...}."""
        tries: list[dict] = []
        stop_dose, stop_reason = None, ""
        while True:
            left = self.max_tries - len(tries)
            decision = self.muse.tune_decide(
                goal, self.patch_label or patch_id, tries, left)
            if decision["action"] == "stop":
                stop_dose = decision["dose"]
                stop_reason = decision["reason"]
                self._emit("stop", dose=stop_dose, reason=stop_reason,
                           by="muse")
                break
            if left <= 0:
                stop_dose = max(tries, key=lambda t: t["gain"])["dose"]
                stop_reason = "try budget used up"
                self._emit("stop", dose=stop_dose, reason=stop_reason,
                           by="harness")
                break
            n = len(tries) + 1
            prediction = {"try": n, "dose": decision["dose"],
                          "expected_gain": decision["expected_gain"],
                          "guard_pass": decision["guard_pass"],
                          "reason": decision["reason"]}
            # The prediction enters the audit log before the try runs.
            self._audit("tune.prediction", patch_id=patch_id, **prediction)
            self._emit("prediction", **prediction)
            got = self.try_fn(decision["dose"])
            confirmed = prediction_confirmed(
                decision["expected_gain"], decision["guard_pass"],
                got["gain"], got["guard_pass"])
            measured = {"try": n, "dose": decision["dose"],
                        "gain": round(float(got["gain"]), 3),
                        "guard_pass": bool(got["guard_pass"]),
                        "guard": got.get("guard"),
                        "confirmed": confirmed,
                        "pairs": got.get("pairs") or [],
                        "seconds": got.get("seconds")}
            self._audit("tune.measured", patch_id=patch_id,
                        expected_gain=decision["expected_gain"],
                        predicted_guard_pass=decision["guard_pass"],
                        **{k: v for k, v in measured.items()
                           if k != "pairs"})
            self._emit("measured", **measured)
            tries.append({**prediction, **measured})

        final = self.final_fn(stop_dose)
        guard = final.get("guard") or {}
        guard_blocked = bool(guard.get("blocked")
                             or guard.get("would_block"))
        gain = round(float(final["gain"]), 3)
        passed = gain >= evaluation.MIN_GAIN and not guard_blocked
        verdict = {"dose": stop_dose, "gain": gain,
                   "min_gain": evaluation.MIN_GAIN,
                   "guard_blocked": guard_blocked, "guard": guard,
                   "passed": passed,
                   "reason": stop_reason, "seconds": final.get("seconds")}
        self._audit("tune.final", patch_id=patch_id, **{
            k: v for k, v in verdict.items() if k != "guard"})
        self._emit("final", **verdict)

        summary_numbers = {
            "goal_numbers": {"tries": len(tries),
                             "max_tries": self.max_tries,
                             "tuning_prompts": TUNING_PROMPTS,
                             "heldout_prompts": HELDOUT_PROMPTS},
            "tries": [{"dose": t["dose"],
                       "expected_gain": t["expected_gain"],
                       "gain": t["gain"],
                       "guard_pass": t["guard_pass"],
                       "confirmed": t["confirmed"]} for t in tries],
            "final": verdict,
            "confirmed": sum(1 for t in tries if t["confirmed"])}
        text = self.muse.tune_summary(summary_numbers)
        self._emit("summary", text=text)
        return {"tune_id": self.id, "goal": goal, "patch_id": patch_id,
                "status": "passed" if passed else "failed",
                "tries": tries, "final": verdict, "summary": text,
                "steps": self.steps,
                "muse": self.muse.info() if hasattr(self.muse, "info")
                else {}}
