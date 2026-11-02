"""MOD phase 7: stage order in one place."""
import re, unittest
from pathlib import Path
from model_lobotomy.create import stages
from model_lobotomy import create_job, trait_eval

class StagesTest(unittest.TestCase):
    def test_order_matches_create_job_and_ui(self):
        self.assertEqual(create_job.STAGES, stages.STAGE_NAMES)
        html=(Path(__file__).resolve().parent.parent/"web"/"js"/"screens"/"workbench.js").read_text()
        ui=re.search(r"const STAGES = \[(.*?)\];", html, re.S).group(1)
        names=re.findall(r'\["(\w+)"', ui)
        self.assertEqual(tuple(names), stages.STAGE_NAMES)
    def test_every_stage_has_impl(self):
        self.assertEqual(tuple(trait_eval.STAGE_FUNCS), stages.STAGE_NAMES)
    def test_pareto_stage_runs_comparison_and_baseline_does_not(self):
        import unittest.mock as mock
        events=[]
        ctx={"pareto_flag": True, "cfg": {"judge": "piratey"}, "fin": ["q"],
             "out": {}, "cancel": None, "progress": lambda *a: events.append(a),
             "params": {"gen_tokens": 8}, "base_cache": object(), "lane": object(),
             "timings": {}, "frozen_layer": 3, "frozen_unit": [0.0], "frozen_dose": 0.5,
             "med": {3: 2.0}, "base": {"top": "Talk like a pirate."}}
        with mock.patch.object(trait_eval.pareto_mod, "run_pareto", return_value={"verdict": "x"}) as run:
            trait_eval._stage_pareto(ctx)
        self.assertEqual(ctx["out"]["pareto"], {"verdict": "x"})
        self.assertIn("pareto", ctx["timings"])
        self.assertEqual(events[0], ("pareto", 0, 1))
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs["instruction"], "Talk like a pirate.")
        with mock.patch.object(trait_eval.pareto_mod, "run_pareto") as run2:
            trait_eval._stage_pareto({**ctx, "pareto_flag": False, "out": {}, "timings": {}})
        run2.assert_not_called()
        import inspect
        self.assertNotIn("run_pareto", inspect.getsource(trait_eval._stage_baseline))
    def test_labels_match_ui(self):
        html=(Path(__file__).resolve().parent.parent/"web"/"js"/"screens"/"workbench.js").read_text()
        for s in stages.STAGES:
            self.assertIn(f'"{s.label}"', html)

if __name__=="__main__": unittest.main()
