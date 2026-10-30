"""E1: evidence and patch routes over a temporary jobs directory (no model)."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

import evidence
import server

LLAMA, QWEN = "meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen3.5-4B"


def card(passed, mean):
    return {"passed": passed, "rules": {}, "trait_mean": mean, "trait_ci": [mean - .1, mean + .1],
            "control_ci": [0, .05], "paired_ci": [.1, .2]}


def make_job(root: Path, jid, trait="hostility", model=LLAMA, status="done", created=1.0, patch=True, report=True):
    d = root / jid
    (d / "trait").mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"id": jid, "trait": trait, "model_id": model, "status": status,
                                            "created": created, "verdict": "knob", "frozen": {"layer": 14, "dose": 0.5}}))
    if report:
        (d / "trait" / "report.json").write_text(json.dumps({
            "trait": trait, "verdict": "knob", "frozen": {"layer": 14, "dose": 0.5},
            "final": {"update_up": {"card": card(True, .4)}, "update_down": {"card": card(False, .1)},
                      "delete": {"card": card(False, .05)}},
            "prompt_baseline": {"card": card(True, .5)},
            "guard": {"blocked": False, "would_block": False, "verified": True, "min_ratio": .9}}))
        (d / "trait" / "report.md").write_text("# Trait E2E\n")
    if patch:
        (d / "trait" / "knob.lobot").write_bytes(b"PK\x05\x06" + b"\0" * 18)
    return d


class EvidenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_job(self.root, "hostility-0-20261030-100000-000", created=1.0)
        make_job(self.root, "hostility-0-20261030-110000-001", created=2.0, patch=False)
        make_job(self.root, "hostility-0-20261030-120000-002", created=3.0, model=QWEN)
        make_job(self.root, "sycophancy-0-20261030-130000-003", trait="sycophancy", status="running", created=4.0, report=False)
        self.p = mock.patch.object(server, "_get_runner", return_value=SimpleNamespace(jobs_dir=self.root))
        self.p.start()
        self.client = TestClient(server.api)

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()

    def test_traits_lists_only_done_jobs(self):
        ids = [t["id"] for t in self.client.get("/api/traits").json()["traits"]]
        self.assertEqual(len(ids), 3)
        self.assertNotIn("sycophancy-0-20261030-130000-003", ids)
        self.assertEqual(ids[0], "hostility-0-20261030-120000-002")  # newest first

    def test_traits_filter_by_model(self):
        t = self.client.get("/api/traits", params={"model_id": LLAMA}).json()["traits"]
        self.assertEqual([x["version"] for x in t], [2, 1])  # versions count per family and model
        self.assertTrue(all(x["model_id"] == LLAMA for x in t))

    def test_trait_item_shape(self):
        t = self.client.get("/api/traits", params={"model_id": LLAMA}).json()["traits"][1]
        self.assertEqual(set(t), {"id", "family", "model_id", "version", "verdict", "layer", "dose",
                                  "created", "cards", "guard", "patch"})
        self.assertEqual(set(t["cards"]), {"update_up", "update_down", "delete", "prompt_baseline"})
        self.assertEqual(set(t["cards"]["update_up"]), {"passed", "trait_mean", "trait_ci"})
        self.assertEqual(t["guard"], {"blocked": False, "would_block": False, "verified": True})
        self.assertEqual(t["patch"], "/api/traits/hostility-0-20261030-100000-000/patch")

    def test_report_json_and_md_served(self):
        jid = "hostility-0-20261030-100000-000"
        r = self.client.get(f"/api/traits/{jid}/report")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["verdict"], "knob")
        md = self.client.get(f"/api/traits/{jid}/report.md")
        self.assertEqual(md.status_code, 200)
        self.assertTrue(md.headers["content-type"].startswith("text/markdown"))

    def test_patch_download_headers(self):
        r = self.client.get("/api/traits/hostility-0-20261030-100000-000/patch")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "application/zip")
        self.assertIn('filename="hostility-v1.lobot"', r.headers["content-disposition"])

    def test_patch_null_when_missing(self):
        t = {x["id"]: x for x in self.client.get("/api/traits").json()["traits"]}
        self.assertIsNone(t["hostility-0-20261030-110000-001"]["patch"])
        self.assertEqual(self.client.get("/api/traits/hostility-0-20261030-110000-001/patch").status_code, 404)

    def test_bad_ids_404(self):
        (self.root.parent / "secret").mkdir(exist_ok=True)
        for bad in ("..", "..%2Fsecret", "%2Fetc%2Fpasswd", "nope", "sycophancy-0-20261030-130000-003", ".hidden"):
            for tail in ("report", "report.md", "patch"):
                self.assertEqual(self.client.get(f"/api/traits/{bad}/{tail}").status_code, 404, (bad, tail))
        self.assertIsNone(evidence.job_dir(self.root, "../" + self.root.name))
        self.assertIsNone(evidence.job_dir(self.root, "/etc"))


if __name__ == "__main__":
    unittest.main()
