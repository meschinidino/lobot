"""MOD phase 9: frontend modules, no build step."""
import re, subprocess, unittest
from pathlib import Path
WEB=Path(__file__).resolve().parent.parent/"web"

class FrontendTest(unittest.TestCase):
    def test_modules_exist_and_index_loads_app(self):
        html=(WEB/"index.html").read_text()
        self.assertIn('src="js/app.js"', html)
        self.assertIn('href="css/app.css"', html)
        for rel in ("js/api.js","js/state.js","js/components/dial.js","js/components/chart.js","js/components/legend.js","js/components/strip.js","js/screens/workbench.js","js/screens/evidence.js","js/screens/library.js","js/screens/new-behavior.js","js/screens/agent.js","js/screens/labeling.js","js/screens/chat.js"):
            self.assertTrue((WEB/rel).is_file(), rel)
    def test_js_syntax(self):
        for fp in sorted((WEB/"js").rglob("*.js")):
            r=subprocess.run(["node","--check",str(fp)], capture_output=True, text=True)
            self.assertEqual(r.returncode,0, f"{fp}: {r.stderr}")
    def test_static_demo_js_syntax(self):
        for fp in sorted((WEB.parent/"deploy"/"static").glob("*.js")):
            r=subprocess.run(["node","--check",str(fp)], capture_output=True, text=True)
            self.assertEqual(r.returncode,0, f"{fp}: {r.stderr}")
    def test_run_error_survives_state_refresh(self):
        # end() refreshes the state and re-renders the status, so the error must be kept per setup.
        run=(WEB/"js"/"screens"/"run.js").read_text()
        bench=(WEB/"js"/"screens"/"workbench.js").read_text()
        self.assertIn("S.runErr = { key: setupKey()", run)
        self.assertIn('S.runErr && S.runErr.key === setupKey()', bench)
    def test_guard_key_shows_both_steps(self):
        run=(WEB/"js"/"screens"/"run.js").read_text()
        self.assertIn("FIRST-TOKEN CHECK PASSED · FULL CHECK RUNNING", run)
        self.assertIn("CHECKING EDITS · FIRST-TOKEN CHECK", run)
        self.assertIn("CHECK PASSED", run)
        self.assertIn("guard_full", run)
    def test_no_knob_explains_why(self):
        ev=(WEB/"js"/"screens"/"evidence.js").read_text()
        pages=(WEB/"js"/"screens"/"pages.js").read_text()
        self.assertIn("whyNoKnob", ev)
        self.assertIn("It was not possible to achieve this behavior "
                      "with a patch.", ev)
        self.assertIn("whyNoKnob(R)", pages)
    def test_agent_run_page_names_its_model(self):
        # A failed run on Qwen next to a sidebar switched to Llama
        # read as a model-selection bug. The run page must name the
        # model the run itself used.
        ag=(WEB/"js"/"screens"/"agent.js").read_text()
        self.assertIn("Run model:", ag)
        self.assertIn("a.model_id", ag)
    def test_no_knob_usable_in_research_mode(self):
        wb=(WEB/"js"/"screens"/"workbench.js").read_text()
        self.assertIn('S.research ? ["warn", "NO KNOB", true]', wb)
    def test_evidence_text_states_only_what_was_measured(self):
        # WHY NO KNOB must name the real scope: candidate layers on
        # tuning prompts. It must not invent a fluency cause the
        # sweep record does not store, and must not say "proves".
        ev=(WEB/"js"/"screens"/"evidence.js").read_text()
        self.assertIn("candidate layer", ev)
        self.assertIn("on tuning prompts", ev)
        self.assertNotIn("broke at least one answer", ev)
        self.assertNotIn("This proves", ev)
        # The tuning grid must apply the p10 gate for v3 and must
        # not present NLL as a gate when the rule has none.
        self.assertIn("fluency_p10", ev)
        self.assertIn("TAIL TOO LOW", ev)
    def test_dose_rule_text_handles_new_rules(self):
        # fluency-first-v2/v3 record no nll_max_rise: the evidence
        # page must not print "undefined" for it.
        ev=(WEB/"js"/"screens"/"evidence.js").read_text()
        self.assertIn("rule.nll_max_rise != null", ev)
        self.assertIn("fluency_p10_floor", ev)
    def test_guard_progress_bar(self):
        run=(WEB/"js"/"screens"/"run.js").read_text()
        html=(WEB/"index.html").read_text()
        self.assertIn("guard_progress", run)
        self.assertIn("guardBarFill", run)
        self.assertIn('id="guardBar"', html)
    def test_reload_warns_while_a_run_is_live(self):
        # The server cancels a run when its stream closes, and results save
        # only on the complete event: a mid-run reload must ask first.
        run=(WEB/"js"/"screens"/"run.js").read_text()
        self.assertIn("beforeunload", run)
        self.assertIn("if (S.running) e.preventDefault()", run)
    def test_no_inline_script_or_style(self):
        html=(WEB/"index.html").read_text()
        self.assertNotIn("<style>", html)
        self.assertNotIn("const STAGES", html)

if __name__=="__main__": unittest.main()
