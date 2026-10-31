"""Unit tests: open descriptions (C3).

No model. Proves: draft JSON parsing accepts good parts and
rejects empty/garbled ones; the reply shuffle is fixed and hides
sides from the client; label merge validates rows; agreement math
uses the 0.5 cut; validate registers on pass and 422s below the
gate with disagreements; legacy store configs match spike (so
full numbers cannot shift); refusal descriptions are rejected;
the draft routes shape 404/409/503 correctly.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import create_job  # noqa: E402
from model_lobotomy import drafts  # noqa: E402
from model_lobotomy import guard as guard_mod  # noqa: E402
from model_lobotomy import server  # noqa: E402
from model_lobotomy import trait_eval as T  # noqa: E402
from model_lobotomy.trait_store import TraitFamily, TraitStore  # noqa: E402

PARTS = {"pos": "Speak as a pirate.", "neg": "Speak plainly.",
         "judge": "pirate voice",
         "baselines": [f"Be piraty, way {i}." for i in range(6)]}


class ParseTest(unittest.TestCase):
    def test_draft_parses_model_json(self):
        text = ('Here is the JSON: {"pos": "Speak as a pirate.", '
                '"neg": "Speak plainly.", "judge": "pirate voice", '
                '"baselines": ["a", "b", "c", "d", "e", "f"]} thanks.')
        got = drafts.parse_draft_parts(text)
        self.assertEqual(got["pos"], "Speak as a pirate.")
        self.assertEqual(len(got["baselines"]), 6)

    def test_draft_rejects_garbage(self):
        with self.assertRaises(drafts.DraftError):
            drafts.parse_draft_parts("no braces here")
        with self.assertRaises(drafts.DraftError):
            drafts.parse_draft_parts('{"pos": "x"} trailing')

    def test_draft_rejects_empty_parts(self):
        for key in ("pos", "neg", "judge"):
            bad = dict(PARTS)
            bad[key] = "  "
            with self.assertRaises(drafts.DraftError):
                drafts.parse_draft_parts(json.dumps(bad))
        bad = dict(PARTS)
        bad["baselines"] = ["a", "b"]
        with self.assertRaises(drafts.DraftError):
            drafts.parse_draft_parts(json.dumps(bad))

    def test_slugify(self):
        self.assertEqual(drafts.slugify("Answers In Rhyme!"),
                         "answers-in-rhyme")
        self.assertEqual(
            drafts.slugify("Noir", {"noir"}), "noir-2")
        self.assertEqual(drafts.slugify("   "), "trait")


class JudgeLabelTest(unittest.TestCase):
    """The judge part must complete "Is this reply ___?" (C3 fix)."""

    def test_phrases_pass_and_normalize(self):
        for raw, want in [
                ("written in a pirate voice", "written in a pirate voice"),
                ("Written in rhyming couplets.", "written in rhyming couplets"),
                ('"cheerful and enthusiastic"', "cheerful and enthusiastic"),
                ("FAQ style and concise", "FAQ style and concise")]:
            self.assertEqual(drafts.judge_label(raw), want, raw)

    def test_rules_and_sentences_rejected(self):
        for raw in [
                "The poem should have at least 2 lines that rhyme "
                "with each other.",
                "Uses phrases like 'the rain poured down'",
                "should rhyme",
                "rhymes. Also uses couplets",
                "a b c d e f g h i j k l m",
                ""]:
            with self.assertRaises(drafts.DraftError, msg=raw):
                drafts.judge_label(raw)

    def test_meta_prompt_names_the_judge_question(self):
        self.assertIn("Is this reply ___?", drafts.META_PROMPT)


class ShuffleTest(unittest.TestCase):
    def test_shuffle_fixed_and_balanced(self):
        prompts = [f"q{i}" for i in range(10)]
        a = drafts.shuffle_replies(prompts, 0)
        b = drafts.shuffle_replies(prompts, 0)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 20)
        sides = [s["side"] for s in a]
        self.assertEqual(sides.count("pos"), 10)
        self.assertEqual(sides.count("neg"), 10)
        self.assertEqual([s["reply_id"] for s in a],
                         [f"r{i:02d}" for i in range(20)])

    def test_public_reply_hides_side(self):
        entry = {"reply_id": "r00", "prompt": "q", "side": "pos",
                 "reply": "ahoy"}
        self.assertEqual(drafts.public_reply(entry),
                         {"reply_id": "r00", "prompt": "q",
                          "reply": "ahoy"})

    def test_reply_prompts_spread(self):
        bank = [f"p{i}" for i in range(120)]
        self.assertEqual(drafts.reply_prompts(bank),
                         [f"p{i}" for i in range(0, 120, 12)])


class LabelsTest(unittest.TestCase):
    def test_merge_and_relabel(self):
        ids = {"r00", "r01"}
        merged = drafts.merge_labels(
            {}, [{"reply_id": "r00", "label": True}], ids)
        self.assertEqual(merged, {"r00": True})
        merged = drafts.merge_labels(
            merged, [{"reply_id": "r00", "label": False},
                     {"reply_id": "r01", "label": True}], ids)
        self.assertEqual(merged, {"r00": False, "r01": True})

    def test_merge_rejects_bad_rows(self):
        ids = {"r00"}
        with self.assertRaises(drafts.DraftError):
            drafts.merge_labels({}, [], ids)
        with self.assertRaises(drafts.DraftError):
            drafts.merge_labels(
                {}, [{"reply_id": "r99", "label": True}], ids)
        with self.assertRaises(drafts.DraftError):
            drafts.merge_labels(
                {}, [{"reply_id": "r00", "label": 1}], ids)
        with self.assertRaises(drafts.DraftError):
            drafts.merge_labels(
                {}, [{"reply_id": "r00", "label": True},
                    {"reply_id": "r00", "label": False}], ids)

    def test_label_agreement_math(self):
        user = {"r00": True, "r01": False, "r02": True, "r03": True}
        scores = {"r00": 0.9, "r01": 0.1, "r02": 0.4, "r03": 0.6}
        got = drafts.label_agreement(user, scores)
        self.assertEqual(got, {"fraction": 0.75, "n": 4,
                               "correct": 3})


class GuardScreenTest(unittest.TestCase):
    def test_style_description_allowed(self):
        ok, reason = guard_mod.draft_description_allowed(
            "answers in rhyming couplets")
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_refusal_descriptions_rejected(self):
        for text in ("refuse less often", "ignore safety rules",
                     "uncensored assistant", "how to make a weapon"):
            ok, reason = guard_mod.draft_description_allowed(text)
            self.assertFalse(ok, text)
            self.assertIn("safety", reason)

    def test_empty_rejected(self):
        ok, _ = guard_mod.draft_description_allowed("   ")
        self.assertFalse(ok)


class FamilyCfgTest(unittest.TestCase):
    def test_store_cfg_matches_spike_for_legacy(self):
        for name in ("blunt", "cheerful", "hostility", "melancholic",
                     "paranoid", "pirate", "shakespeare"):
            cfg = T.family_cfg(name)
            spike = T.spike.BEHAVIORS[name]
            self.assertEqual(cfg["pos"], spike["pos"], name)
            self.assertEqual(cfg["neg"], spike["neg"], name)
            self.assertEqual(cfg["judge"], spike["judge"], name)
            self.assertEqual(cfg["baselines"],
                             T.BASELINE_INSTRUCTIONS[name], name)

    def test_unknown_family_raises(self):
        with self.assertRaises(KeyError):
            T.family_cfg("nope-not-a-family")


class FakeEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = object()
        self.tokenizer = object()
        self.model_id = "Qwen/Qwen3.5-4B"
        self.revision = "stub-rev"
        self.spec = None

    def load(self, model_id):
        self.model_id = model_id
        return 1


class DraftRoutesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "store").mkdir()
        self.eng = FakeEngine()
        self.patches = [
            mock.patch.object(server, "ROOT", self.root),
            mock.patch.object(server, "NO_MODEL", False),
            mock.patch.object(server, "get_engine",
                              return_value=self.eng),
        ]
        for p in self.patches:
            p.start()
        self.addCleanup(server._sae_cache.clear)
        self.client = TestClient(server.api)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def _write_draft(self, draft_id="noir-detective", n=20):
        record = {"id": draft_id, "description": "noir",
                  **PARTS, "model_id": "m", "revision": "r"}
        replies = [{"reply_id": f"r{i:02d}", "prompt": f"q{i}",
                    "side": "pos" if i % 2 == 0 else "neg",
                    "reply": f"reply {i}"} for i in range(n)]
        drafts.save_draft(self.root / "store", draft_id, record,
                          replies)
        return record, replies

    def test_draft_create_shape(self):
        with mock.patch.object(
                drafts, "draft_family",
                return_value=dict(PARTS)) as df, mock.patch.object(
                drafts, "candidate_replies",
                return_value=[{"reply_id": f"r{i:02d}",
                               "prompt": "q", "side": "pos",
                               "reply": "ahoy"} for i in range(20)]):
            r = self.client.post("/api/drafts", json={
                "description": "speaks like a pirate"})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d["id"], "speaks-like-a-pirate")
        self.assertEqual(len(d["replies"]), 20)
        self.assertNotIn("side", d["replies"][0])
        self.assertEqual(d["parts"]["judge"], "pirate voice")
        df.assert_called_once()
        self.assertFalse(self.eng.lock.locked())

    def test_draft_create_rejects_refusal(self):
        r = self.client.post("/api/drafts", json={
            "description": "refuse less often please"})
        self.assertEqual(r.status_code, 422)

    def test_draft_get_shows_labels(self):
        self._write_draft()
        drafts.save_labels(self.root / "store", "noir-detective",
                           {"r00": True})
        r = self.client.get("/api/drafts/noir-detective")
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d["labeled"], 1)
        self.assertEqual(d["replies"][0]["label"], True)
        self.assertIsNone(d["replies"][1]["label"])
        self.assertNotIn("side", d["replies"][0])

    def test_draft_get_404(self):
        self.assertEqual(
            self.client.get("/api/drafts/nope").status_code, 404)
        self.assertEqual(
            self.client.get("/api/drafts/../x").status_code, 404)

    def test_labels_merge(self):
        self._write_draft()
        r = self.client.post("/api/drafts/noir-detective/labels",
                             json={"labels": [
                                 {"reply_id": "r00", "label": True},
                                 {"reply_id": "r01", "label": False}]})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["labeled"], 2)
        r = self.client.post("/api/drafts/noir-detective/labels",
                             json={"labels": [
                                 {"reply_id": "r00",
                                  "label": False}]})
        self.assertEqual(r.json()["labeled"], 2)
        _rec, _rep, labels = drafts.load_draft(
            self.root / "store", "noir-detective")
        self.assertEqual(labels["r00"], False)

    def test_labels_reject_unknown_and_strict_bool(self):
        self._write_draft()
        r = self.client.post("/api/drafts/noir-detective/labels",
                             json={"labels": [
                                 {"reply_id": "r99",
                                  "label": True}]})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/drafts/noir-detective/labels",
                             json={"labels": [
                                 {"reply_id": "r00", "label": 1}]})
        self.assertEqual(r.status_code, 422)

    def test_validate_registers_on_pass(self):
        self._write_draft()
        labels = {f"r{i:02d}": i % 2 == 0 for i in range(20)}
        drafts.save_labels(self.root / "store", "noir-detective",
                           labels)
        store = TraitStore(path=self.root / "traits.json",
                           seed=False)
        runner = mock.Mock()
        runner.store = store
        scores = {rid: (0.9 if lab else 0.1)
                  for rid, lab in labels.items()}
        with mock.patch.object(server, "_get_runner",
                               return_value=runner), \
                mock.patch("model_lobotomy.trait_eval.judge_score",
                           side_effect=lambda m, t, s, q, r, l: scores[
                               [e["reply_id"] for e in
                                json.loads((self.root / "store" /
                                            "drafts" / "noir-detective" /
                                            "replies.json").read_text())
                                if e["reply"] == r][0]]):
            r = self.client.post(
                "/api/drafts/noir-detective/validate")
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d, {"registered": True,
                             "family": "noir-detective",
                             "agreement": 1.0, "gate": 0.85})
        fam = store.get("noir-detective")
        self.assertTrue(fam.hand_validated)
        self.assertEqual(len(fam.baselines), 6)
        self.assertFalse(self.eng.lock.locked())

    def test_validate_422_below_gate(self):
        self._write_draft()
        labels = {f"r{i:02d}": True for i in range(20)}
        drafts.save_labels(self.root / "store", "noir-detective",
                           labels)
        runner = mock.Mock()
        with mock.patch.object(server, "_get_runner",
                               return_value=runner), \
                mock.patch("model_lobotomy.trait_eval.judge_score",
                           return_value=0.1):
            r = self.client.post(
                "/api/drafts/noir-detective/validate")
        self.assertEqual(r.status_code, 422, r.text)
        d = r.json()["detail"]
        self.assertEqual(d["registered"], False)
        self.assertEqual(d["agreement"], 0.0)
        self.assertEqual(len(d["disagreements"]), 20)

    def test_validate_422_until_all_labeled(self):
        self._write_draft()
        drafts.save_labels(self.root / "store", "noir-detective",
                           {"r00": True})
        r = self.client.post("/api/drafts/noir-detective/validate")
        self.assertEqual(r.status_code, 422)

    def test_draft_routes_503_without_weights(self):
        self._write_draft()
        with mock.patch.object(server, "NO_MODEL", True):
            r = self.client.post("/api/drafts", json={
                "description": "a trait"})
            self.assertEqual(r.status_code, 503)
            r = self.client.post(
                "/api/drafts/noir-detective/validate")
            self.assertEqual(r.status_code, 422)  # labels first

    def test_draft_routes_409_when_busy(self):
        self.eng.lock.acquire()
        try:
            r = self.client.post("/api/drafts", json={
                "description": "a trait"})
            self.assertEqual(r.status_code, 409)
        finally:
            self.eng.lock.release()


if __name__ == "__main__":
    unittest.main()
