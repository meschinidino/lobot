"""MOD phase 8: routers + AppState."""
import unittest
from model_lobotomy import server

class RoutersTest(unittest.TestCase):
    def test_app_state_delegates(self):
        st=server.api.state.app_state
        self.assertIs(st.cancel, server._cancel)
        st.runner="x"; self.assertEqual(server._runner,"x"); st.runner=None
    def test_all_areas_registered(self):
        from model_lobotomy.routers import run, create, evidence, drafts_agent, chat, install_verify, reader_features
        total=sum(len(m.router.routes) for m in (run,create,evidence,drafts_agent,chat,install_verify,reader_features))
        self.assertEqual(total,51)
    def test_ndjson_shared(self):
        from model_lobotomy import streaming
        self.assertTrue(callable(streaming.ndjson))

if __name__=="__main__": unittest.main()
