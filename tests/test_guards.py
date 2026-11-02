"""MOD phase 5: guard contract for every registered guard."""
import unittest
from model_lobotomy import guard as guard_mod, guards

class Contract:
    def make(self): raise NotImplementedError
    def test_verdict_shape(self):
        v=self.make().screen(None, None, medians={}) if False else self.make().screen(object(), object())
        d=v.to_dict()
        for k in ("blocked","would_block","verified","enforced","research_mode","screens"):
            self.assertIn(k,d)
        self.assertIsInstance(d["screens"],dict)

class NoopTest(Contract, unittest.TestCase):
    def make(self): return guards.NoopGuard()
    def test_noop_never_blocks(self):
        self.assertFalse(guards.NoopGuard().confirm(None,None).blocked)

class ScreensShapeTest(unittest.TestCase):
    def test_from_old_maps_measure(self):
        gv=guard_mod.GuardVerdict(blocked=True, min_ratio=0.1, worst_prompt="p", blocked_prompts=("p",), full_generation=True)
        d=guards.from_old(gv, enforced=True).to_dict()
        self.assertTrue(d["blocked"] and d["verified"] and d["enforced"])
        s=d["screens"]["refusal"]
        self.assertEqual(s["flagged_prompts"],["p"])
        self.assertIn("min_ratio",s["measure"]); self.assertIn("limit",s["measure"])
        self.assertIsInstance(s["measure_text"],str)
    def test_get_guard_registry(self):
        self.assertEqual(guards.get_guard("noop").name,"noop")
        self.assertEqual(guards.get_guard("screens-v2").name,"screens-v2")
        with self.assertRaises(ValueError): guards.get_guard("nope")

if __name__=="__main__": unittest.main()
