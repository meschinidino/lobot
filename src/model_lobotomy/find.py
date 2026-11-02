"""Find with Muse: Muse Spark finds a neuron edit with lobot tools.

Muse gets five tools: find_neurons, read_neuron, measure_lift,
try_recipe and stop. Each tool wraps code that already exists.
The harness enforces the limits in the constants below.

measure_lift and try_recipe need a prediction. The harness writes
that prediction to the audit log before the tool runs. It appends
the result with confirmed true or false. A lift prediction
confirms when the measured lift lands within a factor of 2 of
the prediction. A recipe prediction confirms when at least one
predicted word shows in the edited answer and not in the plain
answer.

At stop, lobot runs the full guard check on the recipe and
measures the lift of the target words on the held-out lift
prompts. Pass needs lift at or over LIFT_BAR and no guard block.
lobot decides pass or fail, not Muse. Muse only writes the
final summary, and it must pass the number check.

The guard stays on. This module never turns it off and never
sets Research mode. A guard block is a result.
"""
from __future__ import annotations

import json
import math
import time
import uuid

from model_lobotomy import audit
from model_lobotomy import evaluation
from model_lobotomy import muse as muse_mod
from model_lobotomy.app import DOSE_MAX, DOSE_MIN

MAX_TOOL_CALLS = 12
MAX_RECIPE_NEURONS = 4
MAX_TURNS = 16

TOOLS = [
    {"type": "function", "name": "find_neurons",
     "description": "Rank neurons that push one word hardest.",
     "parameters": {"type": "object",
                    "properties": {"word": {"type": "string"},
                                   "k": {"type": "integer"}},
                    "required": ["word"]}},
    {"type": "function", "name": "read_neuron",
     "description": "Read one neuron: what it pushes toward and "
                    "what it fires on.",
     "parameters": {"type": "object",
                    "properties": {"layer": {"type": "integer"},
                                   "neuron": {"type": "integer"}},
                    "required": ["layer", "neuron"]}},
    {"type": "function", "name": "measure_lift",
     "description": "Measure the steering lift of one neuron at "
                    "one dose on the held-out lift prompts.",
     "parameters": {"type": "object",
                    "properties": {
                        "layer": {"type": "integer"},
                        "neuron": {"type": "integer"},
                        "dose": {"type": "number"},
                        "words": {"type": "array",
                                  "items": {"type": "string"}},
                        "prediction": {"type": "number"}},
                    "required": ["layer", "neuron", "dose", "words",
                                 "prediction"]}},
    {"type": "function", "name": "try_recipe",
     "description": "Run a recipe of neurons on the user prompt. "
                    "Return the plain and edited answers and the "
                    "fork token, with the guard first-token check.",
     "parameters": {"type": "object",
                    "properties": {
                        "rows": {"type": "array", "items": {
                            "type": "object",
                            "properties": {
                                "layer": {"type": "integer"},
                                "neuron": {"type": "integer"},
                                "dose": {"type": "number"}},
                            "required": ["layer", "neuron", "dose"]}},
                        "prediction": {"type": "array",
                                       "items": {"type": "string"}}},
                    "required": ["rows", "prediction"]}},
    {"type": "function", "name": "stop",
     "description": "Finish with the final recipe, its target "
                    "words, and the reason.",
     "parameters": {"type": "object",
                    "properties": {
                        "rows": {"type": "array", "items": {
                            "type": "object",
                            "properties": {
                                "layer": {"type": "integer"},
                                "neuron": {"type": "integer"},
                                "dose": {"type": "number"}},
                            "required": ["layer", "neuron", "dose"]}},
                        "words": {"type": "array",
                                  "items": {"type": "string"}},
                        "reason": {"type": "string"}},
                    "required": ["rows", "words", "reason"]}},
]

MEASURING = ("measure_lift", "try_recipe")


def lift_confirmed(prediction: float, measured: float) -> bool:
    """True when the measured lift is within a factor of 2."""
    if prediction is None or measured is None:
        return False
    lo, hi = sorted((float(prediction), float(measured)))
    if lo <= 0:
        return lo == hi
    return hi / lo <= 2.0


def recipe_confirmed(words, plain: str, edited: str,
                     fluency) -> bool:
    """True when a predicted word is new and the text stays fluent.

    The word half: a predicted word shows in the edited answer and
    not in the plain one. The fluency half: the edited answer
    scores at or over FLUENCY_FLOOR. Lift alone passes repeated
    words; this check does not.
    """
    from model_lobotomy import trait_eval
    plain_l, edited_l = (plain or "").lower(), (edited or "").lower()
    new_word = any(str(w).lower() in edited_l
                   and str(w).lower() not in plain_l
                   for w in (words or []))
    return bool(new_word and fluency is not None
                and float(fluency) >= trait_eval.FLUENCY_FLOOR)


def _check_rows(rows) -> list:
    """Checked recipe rows. Raises ValueError on any limit break."""
    rows = list(rows or [])
    if len(rows) > MAX_RECIPE_NEURONS:
        raise ValueError(
            f"a recipe holds at most {MAX_RECIPE_NEURONS} neurons")
    out = []
    for r in rows:
        dose = float(r["dose"])
        if not DOSE_MIN <= dose <= DOSE_MAX:
            raise ValueError(
                f"dose {dose:g} outside [{DOSE_MIN:g}, {DOSE_MAX:g}]")
        out.append({"layer": int(r["layer"]), "neuron": int(r["neuron"]),
                    "dose": dose})
    return out


class FindAgent:
    """One find run. tool_fns maps each tool name to its worker.

    tool_fns holds find_neurons, read_neuron, measure_lift and
    try_recipe. final_fn(rows) returns {"guard": {...}}.
    lift_fn(rows, words) returns {"lift": float}. Tests pass
    fakes. The route passes the live versions. progress(kind, row)
    receives each step as it happens.
    """

    def __init__(self, muse, tool_fns, final_fn, lift_fn, *,
                 fluency_fn=None, audit_log=None, progress=None):
        self.muse = muse
        self.tool_fns = tool_fns
        self.final_fn = final_fn
        self.lift_fn = lift_fn
        self.fluency_fn = fluency_fn
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
        audit.record(event, log=self.audit_log, find_id=self.id, **fields)

    def run(self, goal: str, prompt: str) -> dict:
        """Run the tool loop. Returns the result dict."""
        items = [{"role": "user", "content": muse_mod.FIND_PROMPT.format(
            goal=goal.strip(), prompt=prompt.strip())}]
        calls, confirmed_count = 0, 0
        recipes_tried: list = []
        recipes_confirmed = 0
        stop = None
        last_recipe: list = []
        last_words: list = []
        for _turn in range(MAX_TURNS):
            if stop is not None or calls >= MAX_TOOL_CALLS:
                break
            output = self.muse.tool_turn(items, TOOLS)
            calls_out = [it for it in output
                         if it.get("type") == "function_call"]
            if not calls_out:
                items.append({"role": "user", "content":
                              "Call one tool now. Call stop when "
                              "the recipe looks right."})
                continue
            for call in calls_out:
                if calls >= MAX_TOOL_CALLS:
                    break
                calls += 1
                name = call.get("name")
                try:
                    args = json.loads(call.get("arguments") or "{}")
                except ValueError:
                    args = {}
                result, confirmed, done = self._run_tool(
                    name, args, goal, prompt)
                if confirmed:
                    confirmed_count += 1
                if name == "try_recipe" and "error" not in result:
                    try:
                        last_recipe = _check_rows(args.get("rows"))
                        last_words = [str(w) for w in
                                      (args.get("prediction") or [])]
                        recipes_tried.append(last_recipe)
                        recipes_confirmed += 1 if confirmed else 0
                    except ValueError:
                        pass
                items.append({"type": "function_call", "name": name,
                              "arguments": call.get("arguments") or "{}",
                              "call_id": call.get("call_id")})
                items.append({"type": "function_call_output",
                              "call_id": call.get("call_id"),
                              "output": json.dumps(result, default=str)})
                if done is not None:
                    stop = done
                    break
        if stop is None:
            stop = {"rows": last_recipe, "words": last_words,
                    "reason": "tool budget used up"}
            self._emit("stop", by="harness", **stop)
        else:
            self._emit("stop", by="Muse", **stop)

        from model_lobotomy import trait_eval
        rows = stop["rows"]
        guard = (self.final_fn(rows) or {}).get("guard") or {}
        guard_blocked = bool(guard.get("blocked")
                             or guard.get("would_block"))
        lift = None
        if rows and stop["words"]:
            lift = round(float(
                self.lift_fn(rows, stop["words"])["lift"]), 3)
        fluency = None
        if rows and self.fluency_fn is not None:
            fluency = round(float(self.fluency_fn(rows)), 3)
        fluency_ok = bool(fluency is not None
                          and fluency >= trait_eval.FLUENCY_FLOOR)
        passed = bool(lift is not None and lift >= evaluation.LIFT_BAR
                      and fluency_ok and not guard_blocked)
        if guard_blocked:
            verdict_reason = "guard blocked"
        elif lift is None or lift < evaluation.LIFT_BAR:
            verdict_reason = "lift below bar"
        elif not fluency_ok:
            verdict_reason = "fluency below floor"
        else:
            verdict_reason = "all checks passed"
        final = {"rows": rows, "words": stop["words"],
                 "lift": lift, "lift_bar": evaluation.LIFT_BAR,
                 "fluency": fluency,
                 "fluency_floor": trait_eval.FLUENCY_FLOOR,
                 "guard_blocked": guard_blocked, "guard": guard,
                 "passed": passed, "reason": stop["reason"]}
        self._audit("find.final", rows=rows, words=stop["words"],
                    lift=lift, lift_bar=evaluation.LIFT_BAR,
                    fluency=fluency, guard_blocked=guard_blocked,
                    passed=passed)
        self._emit("final", **final)

        summary_numbers = {
            "tool_calls": calls, "max_tool_calls": MAX_TOOL_CALLS,
            "max_recipe_neurons": MAX_RECIPE_NEURONS,
            "confirmed": confirmed_count,
            "tried_recipes": recipes_tried,
            "recipe": rows, "words": len(stop["words"]),
            "final": {k: v for k, v in final.items() if k != "guard"},
            "guard_blocked": guard_blocked}
        count_sentence = (
            f"Muse used {calls} of {MAX_TOOL_CALLS} tool calls and "
            f"tried {len(recipes_tried)} recipes. "
            f"{recipes_confirmed} were confirmed. Final: "
            f"{'PASS' if passed else 'FAIL'} ({verdict_reason}).")
        text = count_sentence + " " + self.muse.find_summary(
            summary_numbers)
        self._emit("summary", text=text)
        return {"find_id": self.id, "goal": goal, "prompt": prompt,
                "status": "passed" if passed else "failed",
                "tool_calls": calls, "final": final, "summary": text,
                "steps": self.steps,
                "muse": self.muse.info() if hasattr(self.muse, "info")
                else {}}

    def _run_tool(self, name, args, goal, prompt):
        """Run one tool. Returns (output, confirmed, stop_or_None)."""
        if name == "stop":
            try:
                rows = _check_rows(args.get("rows"))
            except (ValueError, TypeError, KeyError) as e:
                return {"error": str(e)}, False, None
            return ({"stopped": True}, False,
                    {"rows": rows,
                     "words": [str(w) for w in (args.get("words") or [])],
                     "reason": str(args.get("reason", ""))[:300]})
        if name not in self.tool_fns:
            return {"error": f"unknown tool {name!r}"}, False, None
        try:
            if name in ("try_recipe", "stop"):
                args = dict(args)
                args["rows"] = _check_rows(args.get("rows"))
            if name in ("measure_lift", "try_recipe"):
                dose = args.get("dose")
                if name == "measure_lift" and dose is not None and \
                        not DOSE_MIN <= float(dose) <= DOSE_MAX:
                    raise ValueError(
                        f"dose {dose:g} outside "
                        f"[{DOSE_MIN:g}, {DOSE_MAX:g}]")
        except (ValueError, TypeError, KeyError) as e:
            got = {"error": str(e)}
            self._emit("result", tool=name, args=args, result=got,
                       confirmed=None, seconds=0.0)
            return got, False, None

        prediction = args.get("prediction")
        if name in MEASURING:
            # The prediction enters the audit log before the tool runs.
            self._audit("find.prediction", tool=name,
                        args={k: v for k, v in args.items()})
            self._emit("prediction", tool=name, args=args,
                       prediction=prediction)
        t0 = time.time()
        try:
            got = self.tool_fns[name](args)
        except Exception as e:  # noqa: BLE001 - a tool error is a result
            got = {"error": f"{type(e).__name__}: {e}"}
        seconds = round(time.time() - t0, 2)
        confirmed = False
        if name == "measure_lift" and "lift" in got:
            confirmed = lift_confirmed(prediction, got["lift"])
            got = dict(got)
            got["lift_display"] = f"×{float(got['lift']):.3g}"
            got["log10_lift"] = (round(math.log10(got["lift"]), 3)
                                 if got["lift"] > 0 else None)
        elif name == "try_recipe" and "edited" in got:
            confirmed = recipe_confirmed(
                prediction, got.get("plain", ""), got.get("edited", ""),
                got.get("fluency"))
        if name in MEASURING:
            self._audit("find.result", tool=name, confirmed=confirmed,
                        seconds=seconds,
                        result={k: v for k, v in got.items()
                                if k not in ("plain", "edited")})
        self._emit("result", tool=name, args=args, result=got,
                   confirmed=confirmed if name in MEASURING else None,
                   seconds=seconds)
        return got, confirmed, None
