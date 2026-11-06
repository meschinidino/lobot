"""MOD phase 6: direction + dose-rule contract."""
import unittest, torch
from model_lobotomy import directions, dose_rules

GEN={0.0:{"judge":0.1,"nll":1.0,"fluency":0.9},0.5:{"judge":0.4,"nll":1.5,"fluency":0.9},1.0:{"judge":0.9,"nll":3.0,"fluency":0.9}}
class DoseTest(unittest.TestCase):
    def test_fluency_first_rule_is_default(self):
        r=dose_rules.get_dose_rule()
        self.assertEqual(r.name,"fluency-first-v3")
        self.assertEqual(r.params,{"fluency_floor":0.5,"fluency_p10_floor":0.3})
        # NLL no longer gates: dose 1.0 is fluent, so it wins.
        self.assertEqual(r.coherent(GEN),[0.5,1.0])
        self.assertEqual(r.choose_gen(GEN),1.0)
        broken={0.0:{"judge":0.1,"nll":1.0,"fluency":0.9},1.0:{"judge":0.9,"nll":0.4,"fluency":0.02}}
        self.assertEqual(r.coherent(broken),[])
        self.assertIsNone(r.choose_gen(broken))
        # Mean fluency passes but the worst prompts loop: excluded.
        tail={0.0:{"judge":0.1,"nll":1.0,"fluency":0.9,"fluency_p10":0.9},
              0.5:{"judge":0.4,"nll":1.2,"fluency":0.9,"fluency_p10":0.8},
              1.0:{"judge":0.9,"nll":1.4,"fluency":0.7,"fluency_p10":0.05}}
        self.assertEqual(r.coherent(tail),[0.5])
        self.assertEqual(r.choose_gen(tail),0.5)
        v2=dose_rules.get_dose_rule("fluency-first-v2")
        self.assertEqual(v2.choose_gen(tail),1.0)
    def test_default_rule(self):
        r=dose_rules.get_dose_rule("nll+fluency-v1")
        self.assertEqual(r.name,"nll+fluency-v1"); self.assertEqual(r.params,{"nll_max_rise":1.0,"fluency_floor":0.5})
        self.assertEqual(r.coherent(GEN),[0.5]); self.assertEqual(r.choose_gen(GEN),0.5)
        self.assertEqual(r.choose({3:GEN}), (3,0.5))
    def test_fake_rule(self):
        self.assertEqual(dose_rules.get_dose_rule("fake").choose_gen(GEN),1.0)
    def test_unknown(self):
        with self.assertRaises(ValueError): dose_rules.get_dose_rule("nope")
        with self.assertRaises(ValueError): directions.get_direction("nope")

class FakeEng:
    spec=type("S",(),{"layers":2})()
    def capture(self, texts, layers, point=None, reduce=None):
        return {L: torch.ones(len(texts),4) for L in layers}
class DirectionTest(unittest.TestCase):
    def test_diffmean_derive_and_seps(self):
        d=directions.get_direction("diffmean")
        A,B=d.derive(FakeEng(), FakeEng.spec, {"pos":"p","neg":"n"}, ["q"]*50)
        self.assertIn(0,A)
        s=d.seps(A,B,[0]); self.assertIn("unit",s[0])

if __name__=="__main__": unittest.main()
