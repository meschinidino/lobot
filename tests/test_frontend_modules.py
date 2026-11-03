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
    def test_no_inline_script_or_style(self):
        html=(WEB/"index.html").read_text()
        self.assertNotIn("<style>", html)
        self.assertNotIn("const STAGES", html)

if __name__=="__main__": unittest.main()
