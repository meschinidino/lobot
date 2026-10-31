"""Unit tests: guard screens beyond refusal (GX).

No model. Proves: each screen blocks a planted flip and passes
clean runs; harmful has a prefill fast path; screens combine by
OR; the env flag disables screens; the verdict dict carries the
screens block; splits are disjoint and cover the sets.
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import engine  # noqa: E402
from model_lobotomy import guard  # noqa: E402
from model_lobotomy import guard_screens as S  # noqa: E402


class Tok:
    def encode(self, s, add_special_tokens=False):
        return [0]

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        return "|".join(m["content"] for m in messages) + "|>"


class ScreenEngine:
    """Scripted masses and texts, keyed by (prompt, tag)."""

    def __init__(self, masses=None, texts=None):
        self.masses = masses or {}
        self.texts = texts or {}
        self.tokenizer = Tok()
        self.spec = SimpleNamespace(thinking_kw=None)
        self.calls: list = []

    def _tag(self, edits):
        if edits is None or edits == engine.EMPTY_EDITS:
            return "natural"
        return "steered"

    def prefill(self, prompt, edits=None, medians=None):
        self.calls.append(("prefill", prompt, self._tag(edits)))
        p = self.masses.get((prompt, self._tag(edits)), 0.05)
        p = min(max(p, 1e-6), 1 - 1e-6)
        return torch.tensor([math.log(p / 7)] * 7
                            + [math.log((1 - p) / 993)] * 993)

    def run(self, prompt, edits, medians, max_new_tokens, seed,
            cancel=None, wrapped=None):
        key = wrapped or prompt
        self.calls.append(("run", key, self._tag(edits)))
        return {"text": self.texts[(key, self._tag(edits))]}

    def fingerprint(self):
        return {"rev": "t", "dtype": "fp16"}


def _edits():
    return engine.EditSet(
        neuron=(engine.NeuronEdit(0, 3, 1.0),))


def _masses(prompts, nat, steered):
    return {(p, "natural"): nat for p in prompts} | {
        (p, "steered"): steered for p in prompts}


def _texts(keys, nat, steered):
    return {(k, "natural"): nat for k in keys} | {
        (k, "steered"): steered for k in keys}


class ScreenEnv:
    """Set LOBOT_GUARD_SCREENS for one test, restore after."""

    def __init__(self, case, value):
        self.case = case
        self.value = value

    def __enter__(self):
        self.prev = os.environ.get("LOBOT_GUARD_SCREENS")
        os.environ["LOBOT_GUARD_SCREENS"] = self.value
        return self

    def __exit__(self, *exc):
        if self.prev is None:
            os.environ.pop("LOBOT_GUARD_SCREENS", None)
        else:
            os.environ["LOBOT_GUARD_SCREENS"] = self.prev
        return False


class HarmfulTest(unittest.TestCase):
    def test_harmful_blocks_planted_flip(self):
        g = guard.Guard()
        eng = ScreenEngine(
            texts=_texts(S.HARMFUL_SET, "safe", "harm"))
        v = g.check_screen(
            eng, _edits(), "harmful", medians={},
            max_new_tokens=8, seed=0,
            judge_fn=lambda t: 0.9 if t == "harm" else 0.1)
        self.assertTrue(v.blocked)
        self.assertEqual(len(v.blocked_prompts), 8)
        self.assertFalse([c for c in eng.calls
                          if c[0] == "prefill"])

    def test_harmful_passes_clean(self):
        g = guard.Guard()
        eng = ScreenEngine(
            texts=_texts(S.HARMFUL_SET, "safe", "safe"))
        v = g.check_screen(
            eng, _edits(), "harmful", medians={},
            max_new_tokens=8, seed=0,
            judge_fn=lambda t: 0.1)
        self.assertFalse(v.blocked)
        self.assertFalse(v.would_block)

    def test_harmful_ignores_natural_compliance(self):
        g = guard.Guard()
        eng = ScreenEngine(texts=_texts(S.HARMFUL_SET, "x", "x"))
        v = g.check_screen(eng, _edits(), "harmful", medians={},
                           max_new_tokens=8, seed=0,
                           judge_fn=lambda t: 0.9)
        self.assertFalse(v.blocked)


class HonestyTest(unittest.TestCase):
    def test_honesty_blocks_planted_flip(self):
        g = guard.Guard()
        eng = ScreenEngine(
            texts=_texts(S.HONESTY_SET, "honest", "lies"))
        v = g.check_screen(
            eng, _edits(), "honesty", medians={},
            max_new_tokens=8, seed=0,
            judge_fn=lambda t: 0.9 if t == "honest" else 0.1)
        self.assertTrue(v.blocked)
        self.assertEqual(len(v.blocked_prompts), 8)
        self.assertFalse([c for c in eng.calls
                          if c[0] == "prefill"])

    def test_honesty_passes_clean(self):
        g = guard.Guard()
        eng = ScreenEngine(
            texts=_texts(S.HONESTY_SET, "honest", "honest"))
        v = g.check_screen(
            eng, _edits(), "honesty", medians={},
            max_new_tokens=8, seed=0,
            judge_fn=lambda t: 0.9)
        self.assertFalse(v.blocked)


class PrivacyTest(unittest.TestCase):
    def _engine(self, leak_natural=False, leak_steered=True):
        eng = ScreenEngine()
        for p in S.PRIVACY_PROMPTS:
            wrapped = S.wrap_history(
                eng.tokenizer, eng.spec,
                [*p["history"],
                 {"role": "user", "content": p["user"]}])
            eng.texts[(wrapped, "natural")] = (
                f"the secret is {p['secret']}" if leak_natural
                else "I cannot share that")
            eng.texts[(wrapped, "steered")] = (
                f"the secret is {p['secret']}" if leak_steered
                else "I cannot share that")
        return eng

    def test_privacy_blocks_planted_flip(self):
        g = guard.Guard()
        v = g.check_screen(self._engine(), _edits(), "privacy",
                           medians={}, max_new_tokens=8, seed=0)
        self.assertTrue(v.blocked)
        self.assertEqual(len(v.blocked_prompts), 8)

    def test_privacy_passes_clean(self):
        g = guard.Guard()
        v = g.check_screen(
            self._engine(leak_steered=False), _edits(), "privacy",
            medians={}, max_new_tokens=8, seed=0)
        self.assertFalse(v.blocked)

    def test_privacy_ignores_natural_leak(self):
        g = guard.Guard()
        v = g.check_screen(
            self._engine(leak_natural=True, leak_steered=True),
            _edits(), "privacy", medians={}, max_new_tokens=8,
            seed=0)
        self.assertFalse(v.blocked)

    def test_leaks_matcher(self):
        self.assertTrue(S.leaks("FALCON-42", "it is falcon-42."))
        self.assertFalse(S.leaks("FALCON-42", "I cannot share."))
        self.assertFalse(S.leaks("7491", ""))
        self.assertTrue(S.leaks("12-34-56", "12-34-56"))


class CombineTest(unittest.TestCase):
    def _engine(self):
        texts = _texts(S.HONESTY_SET, "honest", "lies")
        texts.update(_texts(S.HARMFUL_SET, "safe", "safe"))
        eng = ScreenEngine(masses=_masses(S.HARMFUL_SET, 0.05,
                                          0.06),
                           texts=texts)
        for p in S.PRIVACY_PROMPTS:
            wrapped = S.wrap_history(
                eng.tokenizer, eng.spec,
                [*p["history"],
                 {"role": "user", "content": p["user"]}])
            eng.texts[(wrapped, "natural")] = "I cannot share that"
            eng.texts[(wrapped, "steered")] = "I cannot share that"
        return eng

    def test_screens_combine_or(self):
        g = guard.Guard(refusal_set=("zzz?",))
        eng = self._engine()
        eng.masses[("zzz?", "natural")] = 0.9
        eng.masses[("zzz?", "steered")] = 0.88
        eng.texts[("zzz?", "natural")] = "REFUSE"
        eng.texts[("zzz?", "steered")] = "REFUSE"
        judges = {"harmful": lambda t: 0.1,
                  "honesty": lambda t: 0.9 if t == "honest"
                  else 0.1}
        real = guard.Guard.check_screen

        def routed(self, engine, edits, screen, **kw):
            return real(self, engine, edits, screen,
                        judge_fn=judges.get(screen), **kw)

        with ScreenEnv(self, "refusal,harmful,honesty,privacy"), \
                unittest.mock.patch.object(guard.Guard,
                                           "check_screen", routed):
            v = g.check_verified(
                eng, _edits(), medians={}, max_new_tokens=8,
                seed=0, judge_fn=lambda t: 1.0)
        self.assertTrue(v.blocked)
        names = [n for n, _ in v.screens]
        self.assertEqual(names, ["refusal", "harmful", "honesty",
                                 "privacy"])
        by_name = dict(v.screens)
        self.assertFalse(by_name["refusal"].blocked)
        self.assertTrue(by_name["honesty"].blocked)
        d = guard.verdict_dict(v)
        self.assertTrue(d["screens"]["honesty"]["blocked"])
        self.assertFalse(d["screens"]["privacy"]["blocked"])
        self.assertEqual(
            d["screens"]["honesty"]["display_prompt"],
            S.HONESTY_SET[0])

    def test_default_screens_exclude_harmful(self):
        with ScreenEnv(self, ""):
            # Empty selects the default set (harmful opt-in).
            os.environ.pop("LOBOT_GUARD_SCREENS", None)
            self.assertEqual(S.enabled_screens(),
                             ("refusal", "honesty", "privacy"))

    def test_screen_disable_flag(self):
        g = guard.Guard(refusal_set=("zzz?",))
        eng = self._engine()
        eng.masses[("zzz?", "natural")] = 0.9
        eng.masses[("zzz?", "steered")] = 0.88
        eng.texts[("zzz?", "natural")] = "REFUSE"
        eng.texts[("zzz?", "steered")] = "REFUSE"
        with ScreenEnv(self, "refusal"):
            v = g.check_verified(
                eng, _edits(), medians={}, max_new_tokens=8,
                seed=0, judge_fn=lambda t: 1.0)
        self.assertFalse(v.blocked)
        self.assertEqual([n for n, _ in v.screens], ["refusal"])


class SplitsTest(unittest.TestCase):
    def test_screen_splits_disjoint_and_covering(self):
        splits = S.make_screen_splits()
        for screen, full in (("harmful", set(S.HARMFUL_SET)),
                             ("honesty", set(S.HONESTY_SET))):
            tune = set(splits[f"{screen}_tuning"])
            held = set(splits[f"{screen}_held_out"])
            self.assertEqual(tune & held, set())
            self.assertEqual(tune | held, full)
            self.assertEqual(len(held), 2)
        tune = set(splits["privacy_tuning"])
        held = set(splits["privacy_held_out"])
        self.assertEqual(tune & held, set())
        self.assertEqual(tune | held, set(range(8)))
        for screen, names in (
                ("harmful", S.WEAKENING_EDIT_NAMES),
                ("privacy", S.WEAKENING_EDIT_NAMES),
                ("honesty", S.HONESTY_EDIT_NAMES)):
            tune = set(splits[f"{screen}_edits_tuning"])
            held = set(splits[f"{screen}_edits_held_out"])
            self.assertEqual(tune & held, set())
            self.assertEqual(tune | held, set(names))
            self.assertEqual(len(held), 2)

    def test_splits_deterministic(self):
        self.assertEqual(S.make_screen_splits(),
                         S.make_screen_splits())


if __name__ == "__main__":
    unittest.main()
