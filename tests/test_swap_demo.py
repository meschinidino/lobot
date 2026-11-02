"""MOD acceptance 3+4: swap demo on a fake engine, no weights."""
import json, re, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import torch
from model_lobotomy import engine as engine_mod
from model_lobotomy import trait_eval as T

class FakeEng:
    def __init__(self):
        self.model=object(); self.tokenizer=object()
        self.spec=SimpleNamespace(layers=4, hidden=8)
    def fingerprint(self): return {"model_id":"fake"}
    def capture(self, texts, layers, point="block_out", reduce="last"):
        g=torch.Generator().manual_seed(1)
        return {L: torch.randn(len(texts),8, generator=g) for L in layers}
    def run(self, prompt, edits, medians, max_new_tokens, seed=0, cancel=None):
        tag="base" if edits==engine_mod.EMPTY_EDITS else "pirate steered"
        return {"text": f"{tag} reply to {prompt}"}

class SwapDemoTest(unittest.TestCase):
    def test_create_with_swapped_ports(self):
        eng=FakeEng()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(T.spike,"median_norms",return_value=[1.0]*4), \
                 mock.patch.object(T.spike,"nll",return_value=1.0), \
                 mock.patch.dict("os.environ",{"LOBOT_GUARD":"noop","LOBOT_JUDGE":"fake","LOBOT_DOSE_RULE":"fake"}):
                out=T.run_trait(eng, eng.spec, "pirate", [f"q{i}" for i in range(130)], 0, Path(tmp)/"w", "fake", "rev", schedule="fast")
            rep=json.loads((Path(tmp)/"w"/"report.json").read_text())
            self.assertEqual(rep["ports"], {"guard":"noop","judge":"fake","direction":"diffmean","dose_rule":"fake"})
            self.assertEqual(rep["tuning"]["rule"]["name"], "fake")
            self.assertIn(out["verdict"], ("knob","no_knob","blocked"))

    def test_no_search_needed_in_src(self):
        src=Path(__file__).resolve().parent.parent/"src"/"model_lobotomy"
        guard_bad, env_bad, jobs_bad = [], [], []
        for p in src.rglob("*.py"):
            rel=str(p.relative_to(src)); text=p.read_text()
            if "Guard(" in text and rel not in ("guard.py",) and not rel.startswith("guards/"):
                guard_bad.append(rel)
            if re.search(r"os\.environ|os\.getenv", text) and rel!="config.py":
                env_bad.append(rel)
            if re.search(r'"/ "jobs"|"jobs" /|/ "job\.json"', text) and rel!="jobs_store.py":
                jobs_bad.append(rel)
        self.assertEqual(guard_bad, [], guard_bad)
        self.assertEqual(env_bad, [], env_bad)
        self.assertEqual(jobs_bad, [], jobs_bad)

if __name__=="__main__": unittest.main()
