"""PP tests: verdict rule, front, and the stage on a fake lane."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_lobotomy import engine as engine_mod  # noqa: E402
from model_lobotomy import pareto as P  # noqa: E402


def pt(name, trait, cost, kind="vector", hm=0.5):
    return {"name": name, "trait": trait, "cost": cost, "kind": kind,
            "hm": hm}


class VerdictTest(unittest.TestCase):
    def test_vector_dominates(self):
        self.assertEqual(P.verdict([pt("v1", 0.6, 0.1), pt("v2", 0.2, 0.0)],
                                   pt("p", 0.5, 0.1, "prompt")),
                         "vector_dominates")

    def test_prompt_dominates(self):
        self.assertEqual(P.verdict([pt("v1", 0.4, 0.2), pt("v2", 0.1, 0.1)],
                                   pt("p", 0.5, 0.05, "prompt")),
                         "prompt_dominates")

    def test_tradeoff(self):
        # More effect than the prompt, but more cost too.
        self.assertEqual(P.verdict([pt("v1", 0.7, 0.3)],
                                   pt("p", 0.5, 0.1, "prompt")),
                         "tradeoff")

    def test_front_drops_dominated(self):
        pts = [pt("a", 0.5, 0.1), pt("b", 0.4, 0.2), pt("c", 0.7, 0.3)]
        self.assertEqual(P.front(pts), ["a", "c"])

    def test_strongest_prompt(self):
        pts = [pt("v", 0.9, 0.0), pt("prompt", 0.2, 0.0, "prompt"),
               pt("prompt:label", 0.6, 0.1, "prompt")]
        self.assertEqual(P.strongest_prompt(pts)["name"], "prompt:label")

    def test_doses_include_frozen(self):
        self.assertEqual(P.doses_for(1.25), [0.5, 1.0, 1.25, 1.5, 2.0, 2.5])
        self.assertEqual(P.doses_for(1.0), [0.5, 1.0, 1.5, 2.0, 2.5])

    def test_rows_gain_and_cost(self):
        base = {"trait": [0.1], "relevance": [0.9], "fluency": [0.8]}
        rows = P.point_rows(base, {"trait": [0.7], "relevance": [0.5],
                                   "fluency": [0.6]})
        self.assertAlmostEqual(rows[0]["trait"], 0.6)
        self.assertAlmostEqual(rows[0]["cost"], 0.3)
        self.assertGreater(rows[0]["hm"], 0)
        self.assertEqual(P._hm(0.0, 1.0, 1.0), 0.0)


class FakeLane:
    """Replies tag their edits; the judge scores by tag and label."""

    def __init__(self):
        self.gens = []

    def gen_many(self, prompts, edits, max_new, cancel=None):
        self.gens.append((len(prompts), edits))
        if edits.vector:
            tag = f"v{edits.vector[0].dose:g}"
        else:
            tag = "base"
        return [f"{tag}|{q}" for q in prompts]

    def judge_many(self, pairs, label):
        out = []
        for _q, r in pairs:
            tag, q = r.split("|", 1)
            prompted = "\n\n" in q
            if label == P.RELEVANCE_LABEL:
                out.append(0.9 if tag == "base" else
                           max(0.1, 0.9 - 0.2 * float(tag[1:])))
            elif label == P.FLUENCY_LABEL:
                out.append(0.9)
            else:  # trait
                t = 0.1 if tag == "base" else 0.1 + 0.25 * float(tag[1:])
                out.append(min(0.95, t + (0.4 if prompted else 0.0)))
        return out


class RunTest(unittest.TestCase):
    def test_run_pareto_points_and_verdict(self):
        lane = FakeLane()
        final = ["q1?", "q2?", "q3?"]
        block = P.run_pareto(lane, "piratey", final, layer=3,
                             unit=torch.tensor([1.0, 0.0]), scale=2.0,
                             frozen_dose=1.0, instruction="Be a pirate.",
                             max_new=8)
        names = [p["name"] for p in block["points"]]
        self.assertEqual(names, ["vector@0.5", "vector@1", "vector@1.5",
                                 "vector@2", "vector@2.5", "prompt",
                                 "prompt:label", "prompt+vector"])
        # Base + 5 doses + 2 prompts + prompt+vector.
        self.assertEqual(len(lane.gens), 9)
        prompt = next(p for p in block["points"] if p["name"] == "prompt")
        self.assertAlmostEqual(prompt["trait"], 0.4)
        self.assertAlmostEqual(prompt["cost"], 0.0)
        # The prompt gains 0.4 at no cost; every vector point costs.
        self.assertEqual(block["verdict"], "tradeoff")
        self.assertEqual(block["verdict_prompt"], "prompt")
        self.assertIn("ci", block["best_vector"]["trait_minus_prompt"])
        self.assertTrue(P.pareto_lines(block)[0].startswith("Pareto"))
        # Vector edits carry the layer, scale and dose.
        v = lane.gens[1][1].vector[0]
        self.assertEqual((v.layer, v.scale, v.dose), (3, 2.0, 0.5))
        self.assertEqual(lane.gens[0][1], engine_mod.EMPTY_EDITS)


if __name__ == "__main__":
    unittest.main()
