"""MOD phase 3: spike functions live in src, no file loading."""
import re, unittest
from pathlib import Path
from model_lobotomy import spike

class SpikeLibraryTest(unittest.TestCase):
    def test_behaviors_and_dose_math(self):
        self.assertIn("pirate", spike.BEHAVIORS)
        gen={0.0:{"nll":1.0,"judge":0.1},0.5:{"nll":1.5,"judge":0.4},1.0:{"nll":3.0,"judge":0.9}}
        self.assertEqual(spike.coherent_doses(gen),[0.5])
        self.assertEqual(spike.best_dose(gen),0.5)
    def test_no_importlib_file_loading_in_src(self):
        src=Path(__file__).resolve().parent.parent/"src"/"model_lobotomy"
        bad=[p.name for p in src.glob("*.py") if "spec_from_file_location" in p.read_text()]
        self.assertEqual(bad,[])

if __name__=="__main__": unittest.main()
