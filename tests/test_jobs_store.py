"""MOD phase 2: job storage in one place."""
import json, tempfile, unittest
from pathlib import Path
from model_lobotomy import jobs_store

class JobsStoreTest(unittest.TestCase):
    def test_paths(self):
        p = jobs_store.JobPaths("j1", Path("/tmp/jobs"))
        self.assertEqual(str(p.job_json), "/tmp/jobs/j1/job.json")
        self.assertEqual(str(p.report), "/tmp/jobs/j1/trait/report.json")
        self.assertEqual(str(p.patch), "/tmp/jobs/j1/trait/knob.lobot")
        self.assertEqual(str(p.direction), "/tmp/jobs/j1/trait/direction.safetensors")
        self.assertEqual(str(p.pareto), "/tmp/jobs/j1/trait/pareto.json")
        self.assertTrue(str(p.repair("gradient")).endswith("repair_gradient.json"))
        self.assertTrue(str(p.repair("project")).endswith("repair.json"))

    def test_import_and_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp)/"run"; src.mkdir()
            (src/"report.json").write_text(json.dumps({"trait":"pirate","seed":0,"seconds":1,"verdict":"knob","fingerprint":{"model_id":"m","revision":"r"}}))
            (src/"knob.lobot").write_text("x")
            dest = jobs_store.import_run(src, Path(tmp)/"jobs", "p1")
            self.assertTrue((dest/"job.json").is_file())
            self.assertTrue((dest/"trait"/"report.json").is_file())
            self.assertEqual([j["id"] for j in jobs_store.list_jobs(Path(tmp)/"jobs")], ["p1"])
            with self.assertRaises(FileExistsError):
                jobs_store.import_run(src, Path(tmp)/"jobs", "p1")

    def test_workpaths(self):
        w = jobs_store.WorkPaths(Path("/x"))
        self.assertEqual(w.repaired_patch("gradient").name, "knob.repaired-gradient.lobot")

if __name__ == "__main__":
    unittest.main()
