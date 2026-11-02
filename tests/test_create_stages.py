"""MOD phase 7: stage order in one place."""
import re, unittest
from pathlib import Path
from model_lobotomy.create import stages
from model_lobotomy import create_job, trait_eval

class StagesTest(unittest.TestCase):
    def test_order_matches_create_job_and_ui(self):
        self.assertEqual(create_job.STAGES, stages.STAGE_NAMES)
        html=(Path(__file__).resolve().parent.parent/"web"/"index.html").read_text()
        ui=re.search(r"const STAGES = \[(.*?)\];", html, re.S).group(1)
        names=re.findall(r'\["(\w+)"', ui)
        self.assertEqual(tuple(names), stages.STAGE_NAMES)
    def test_every_stage_has_impl(self):
        self.assertEqual(tuple(trait_eval.STAGE_FUNCS), stages.STAGE_NAMES)
    def test_labels_match_ui(self):
        html=(Path(__file__).resolve().parent.parent/"web"/"index.html").read_text()
        for s in stages.STAGES:
            self.assertIn(f'"{s.label}"', html)

if __name__=="__main__": unittest.main()
