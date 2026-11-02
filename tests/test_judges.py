"""MOD phase 4: judge contract for every registered judge."""
import unittest
from unittest import mock
from model_lobotomy import judges, trait_eval

class Contract:
    def make(self): raise NotImplementedError
    def test_scores_in_range_and_shape(self):
        j=self.make()
        out=j.score_many([("q","a pirate reply arr"),("q","plain reply")],"pirate voice")
        self.assertEqual(len(out),2)
        self.assertTrue(all(0.0<=s<=1.0 for s in out))
    def test_memo_dedupes(self):
        j=self.make()
        a=j.score_many([("q","x pirate")],"pirate"); b=j.score_many([("q","x pirate")],"pirate")
        self.assertEqual(a,b)

class FakeTest(Contract, unittest.TestCase):
    def make(self): return judges.FakeJudge()
    def test_fake_sees_label_word(self):
        self.assertEqual(judges.FakeJudge().score_many([("q","arr pirate")],"pirate voice"),[1.0])

class LocalTest(Contract, unittest.TestCase):
    def make(self): return judges.get_judge("local", score_one=lambda q,r,l: 0.25)

class MetaShapeTest(unittest.TestCase):
    def test_registry_and_unknown(self):
        self.assertEqual(judges.REGISTRY,("local","meta_api","fake"))
        with self.assertRaises(ValueError): judges.get_judge("nope")
    def test_meta_is_judge_with_score_many(self):
        j=trait_eval.MetaApiJudge(key="k", model="m", n_samples=1)
        with mock.patch.object(j,"_complete", return_value="Yes"):
            self.assertEqual(j.score_many([("q","reply")],"rubric"),[1.0])

if __name__=="__main__": unittest.main()
