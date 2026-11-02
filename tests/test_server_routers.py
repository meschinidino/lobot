"""MOD phase 8: routers + AppState."""
import unittest

from model_lobotomy import server


class RoutersTest(unittest.TestCase):
    def test_app_state_holds_state_itself(self):
        st = server.api.state.app_state
        self.assertIs(st, server.STATE)
        self.assertIs(st.cancel, server.STATE.cancel)
        old = st.runner
        st.runner = "x"
        self.assertEqual(server.STATE.runner, "x")
        st.runner = old

    def test_all_areas_registered(self):
        from model_lobotomy.routers import run, create, evidence, drafts_agent, chat, install_verify, reader_features
        total = sum(len(m.router.routes) for m in (run, create, evidence, drafts_agent, chat, install_verify, reader_features))
        self.assertEqual(total, 51)

    def test_no_live_or_delegate_shim(self):
        import pathlib
        routers = pathlib.Path(__file__).resolve().parent.parent / "src" / "model_lobotomy" / "routers"
        for p in routers.glob("*.py"):
            text = p.read_text()
            self.assertNotIn("_Live", text, p.name)
            self.assertNotIn("__getattr__", text, p.name)
            self.assertNotIn("__dict__.update", text, p.name)
        self.assertNotIn("_delegate", (routers.parent / "server.py").read_text())

    def test_ndjson_shared(self):
        from model_lobotomy import streaming
        self.assertTrue(callable(streaming.ndjson))


if __name__ == "__main__":
    unittest.main()
