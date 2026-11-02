"""Unit tests: evidence and patch API (E1, server.py).

Value: protects=/api/traits list shape, report/patch serving, patch
headers, id safety; fails_when=the evidence contract drifts or a bad
id escapes jobs/. no_claim=real weights (fixtures only; no model).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import create_job  # noqa: E402
from model_lobotomy import patch as patch_mod  # noqa: E402
from model_lobotomy import server  # noqa: E402

MODEL_A = "Qwen/Qwen3.5-4B"
MODEL_B = "meta-llama/Llama-3.2-3B-Instruct"


def _card(passed=True, mean=0.31, ci=(0.22, 0.40)):
    """Full card as trait_eval writes it (the route keeps 3 fields)."""
    return {"passed": passed, "rules": {"trait_ci_excludes_zero": True,
                                       "trait_mean_ge_min": True,
                                       "control_upper_lt_max": True,
                                       "paired_ci_excludes_zero": True},
            "trait_ci": list(ci), "trait_mean": mean,
            "control_ci": [0.01, 0.05], "paired_ci": [0.2, 0.4]}


def _write_job(jobs_dir: Path, job_id: str, *, status="done",
               trait="hostility", model_id=MODEL_A, created=1793300000.0,
               verdict="knob", frozen=None, with_patch=True,
               with_md=True, with_report=True):
    job_dir = jobs_dir / job_id
    trait_dir = job_dir / "trait"
    trait_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(json.dumps({
        "id": job_id, "trait": trait, "seed": 0, "model_id": model_id,
        "revision": "rev", "research_mode": False, "guard_opt_in": False,
        "status": status, "stage": "verdict",
        "progress": {"stage": "verdict", "done": 1, "total": 1},
        "verdict": verdict if status == "done" else None,
        "frozen": frozen or {}, "eligibility": {}, "guard": {},
        "error": "", "created": created, "updated": created + 1}))
    if with_report:
        report = {"trait": trait, "seed": 0, "verdict": verdict,
                  "frozen": frozen,
                  "final": {"update_up": {"card": _card()},
                            "update_down": {"card": _card(False, 0.05)},
                            "delete": {"card": _card()}},
                  "prompt_baseline": {"card": _card()},
                  "guard": {"blocked": False, "would_block": False,
                            "enforced": True, "research_mode": False,
                            "verified": True, "min_ratio": 0.8,
                            "worst_prompt": None, "blocked_prompts": []},
                  "seconds": 12.5}
        (trait_dir / "report.json").write_text(json.dumps(report))
    if with_md:
        (trait_dir / "report.md").write_text(f"# Trait E2E: {trait}\n")
    if with_patch:
        patch_mod.save_patch(
            trait_dir / "knob.lobot", direction=torch.ones(8),
            layer=14, dose=0.5, trait=trait, recipe={}, report_card={},
            guard={}, fingerprint={}, scale=2.0)
    return job_dir


class TraitsApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.jobs_dir = Path(self.tmp.name) / "jobs"
        self.jobs_dir.mkdir()
        server._runner = create_job.CreateRunner(jobs_dir=self.jobs_dir)
        self.addCleanup(setattr, server, "_runner", None)
        self.client = TestClient(server.api)

    def test_traits_lists_only_done_jobs(self):
        _write_job(self.jobs_dir, "hostility-0-20260101-000000-000",
                   status="done", created=1793300000.0)
        _write_job(self.jobs_dir, "hostility-0-20260101-000001-001",
                   status="running", created=1793300001.0)
        _write_job(self.jobs_dir, "hostility-0-20260101-000002-002",
                   status="failed", created=1793300002.0)
        _write_job(self.jobs_dir, "hostility-0-20260101-000003-003",
                   status="queued", created=1793300003.0)
        r = self.client.get("/api/traits")
        self.assertEqual(r.status_code, 200, r.text)
        ids = [t["id"] for t in r.json()["traits"]]
        self.assertEqual(ids, ["hostility-0-20260101-000000-000"])

    def test_traits_filter_by_model(self):
        _write_job(self.jobs_dir, "hostility-0-20260101-000000-000",
                   model_id=MODEL_A, created=1793300000.0)
        _write_job(self.jobs_dir, "hostility-0-20260101-000001-001",
                   model_id=MODEL_B, created=1793300001.0)
        r = self.client.get("/api/traits", params={"model_id": MODEL_B})
        self.assertEqual(r.status_code, 200, r.text)
        traits = r.json()["traits"]
        self.assertEqual(len(traits), 1)
        self.assertEqual(traits[0]["model_id"], MODEL_B)
        r = self.client.get("/api/traits", params={"model_id": "nope"})
        self.assertEqual(r.json()["traits"], [])

    def test_trait_item_shape(self):
        job_id = "hostility-0-20260101-000000-000"
        _write_job(self.jobs_dir, job_id, frozen={"layer": 14, "dose": 0.5})
        (item,) = self.client.get("/api/traits").json()["traits"]
        self.assertEqual(set(item), {"id", "family", "model_id", "version",
                                    "verdict", "layer", "dose", "created",
                                    "cards", "guard", "patch",
                                    "delete_eligible", "evidence"})
        self.assertEqual(item["id"], job_id)
        self.assertEqual(item["family"], "hostility")
        self.assertEqual(item["model_id"], MODEL_A)
        self.assertEqual(item["version"], 1)
        self.assertEqual(item["verdict"], "knob")
        self.assertEqual(item["layer"], 14)
        self.assertEqual(item["dose"], 0.5)
        self.assertIs(item["delete_eligible"], False)
        self.assertEqual(item["created"], 1793300000.0)
        self.assertEqual(set(item["cards"]),
                         {"update_up", "update_down", "delete",
                          "prompt_baseline"})
        for card in item["cards"].values():
            self.assertEqual(set(card), {"passed", "trait_mean",
                                        "trait_ci"})
        self.assertEqual(item["cards"]["update_up"],
                         {"passed": True, "trait_mean": 0.31,
                          "trait_ci": [0.22, 0.40]})
        self.assertEqual(item["guard"], {"blocked": False,
                                        "would_block": False,
                                        "verified": True})
        self.assertEqual(item["patch"], f"/api/traits/{job_id}/patch")

    def test_traits_sorted_newest_first(self):
        _write_job(self.jobs_dir, "hostility-0-20260101-000000-000",
                   created=1793300000.0)
        _write_job(self.jobs_dir, "hostility-0-20260101-000001-001",
                   created=1793300005.0)
        ids = [t["id"]
               for t in self.client.get("/api/traits").json()["traits"]]
        self.assertEqual(ids, ["hostility-0-20260101-000001-001",
                              "hostility-0-20260101-000000-000"])

    def test_versions_count_runs_per_family_and_model(self):
        _write_job(self.jobs_dir, "hostility-0-20260101-000000-000",
                   created=1793300000.0)
        _write_job(self.jobs_dir, "hostility-0-20260101-000001-001",
                   created=1793300001.0)
        _write_job(self.jobs_dir, "blunt-0-20260101-000002-002",
                   trait="blunt", created=1793300002.0)
        items = {t["id"]: t
                 for t in self.client.get("/api/traits").json()["traits"]}
        self.assertEqual(items["hostility-0-20260101-000000-000"]
                         ["version"], 1)
        self.assertEqual(items["hostility-0-20260101-000001-001"]
                         ["version"], 2)
        self.assertEqual(items["blunt-0-20260101-000002-002"]
                         ["version"], 1)

    def test_no_knob_item_nulls_missing_fields(self):
        job_id = "hostility-0-20260101-000000-000"
        job_dir = _write_job(self.jobs_dir, job_id, verdict="no_knob",
                             frozen=None, with_patch=False)
        report = {"trait": "hostility", "seed": 0, "verdict": "no_knob",
                  "frozen": None, "seconds": 3.0}
        (job_dir / "trait" / "report.json").write_text(
            json.dumps(report))
        (item,) = self.client.get("/api/traits").json()["traits"]
        self.assertEqual(item["verdict"], "no_knob")
        self.assertIsNone(item["layer"])
        self.assertIsNone(item["dose"])
        self.assertEqual(item["cards"], {"update_up": None,
                                        "update_down": None,
                                        "delete": None,
                                        "prompt_baseline": None})
        self.assertIsNone(item["guard"])
        self.assertIsNone(item["patch"])

    def test_report_json_and_md_served(self):
        job_id = "hostility-0-20260101-000000-000"
        job_dir = _write_job(self.jobs_dir, job_id,
                             frozen={"layer": 14, "dose": 0.5})
        want = json.loads((job_dir / "trait" / "report.json").read_text())
        r = self.client.get(f"/api/traits/{job_id}/report")
        self.assertEqual(r.status_code, 200, r.text)
        got = r.json()
        # U4: the report gains pareto, repair, tuning.chosen, followups.
        for key in ("pareto", "repair", "followups"):
            self.assertIn(key, got)
        self.assertIsNone(got.pop("pareto"))
        self.assertIsNone(got.pop("repair"))
        got.pop("followups")
        self.assertIsNone(got.pop("tuning")["chosen"])
        self.assertEqual(got, want)
        r = self.client.get(f"/api/traits/{job_id}/report.md")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.headers["content-type"].startswith(
            "text/markdown"), r.headers["content-type"])
        self.assertEqual(r.text, "# Trait E2E: hostility\n")

    def test_patch_download_headers(self):
        job_id = "hostility-0-20260101-000000-000"
        job_dir = _write_job(self.jobs_dir, job_id,
                             frozen={"layer": 14, "dose": 0.5})
        r = self.client.get(f"/api/traits/{job_id}/patch")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.headers["content-type"].startswith(
            "application/zip"), r.headers["content-type"])
        self.assertIn("attachment",
                      r.headers["content-disposition"])
        self.assertIn("hostility-v1.lobot",
                      r.headers["content-disposition"])
        self.assertEqual(r.content,
                         (job_dir / "trait" / "knob.lobot").read_bytes())
        loaded = patch_mod.load_patch(job_dir / "trait" / "knob.lobot")
        self.assertEqual(loaded["meta"]["trait"], "hostility")

    def test_patch_null_when_missing(self):
        job_id = "hostility-0-20260101-000000-000"
        _write_job(self.jobs_dir, job_id, with_patch=False)
        (item,) = self.client.get("/api/traits").json()["traits"]
        self.assertIsNone(item["patch"])
        r = self.client.get(f"/api/traits/{job_id}/patch")
        self.assertEqual(r.status_code, 404)

    def test_bad_ids_404(self):
        job_id = "hostility-0-20260101-000000-000"
        _write_job(self.jobs_dir, job_id)
        bad = ["no-such-job", "..", ".", "../x", "/abs/path", "a/b",
               "hostility-0-20260101-000000-000."]
        for bid in bad:
            with self.subTest(bid=bid):
                self.assertEqual(
                    self.client.get(f"/api/traits/{bid}/report")
                    .status_code, 404)
                self.assertEqual(
                    self.client.get(f"/api/traits/{bid}/report.md")
                    .status_code, 404)
                self.assertEqual(
                    self.client.get(f"/api/traits/{bid}/patch")
                    .status_code, 404)
        # Encoded traversal hits the helper, not the router.
        for bid in ["..%2F..%2Fetc", "%2E%2E%2F"]:
            with self.subTest(bid=bid):
                self.assertEqual(
                    self.client.get(f"/api/traits/{bid}/report")
                    .status_code, 404)
        # The helper itself rejects traversal and unknown ids.
        from fastapi import HTTPException
        for bid in ["../x", "/abs", "a/b", "", "nope"]:
            with self.subTest(helper=bid):
                with self.assertRaises(HTTPException) as ctx:
                    server._trait_job_dir(bid)
                self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(server._trait_job_dir(job_id),
                         self.jobs_dir / job_id)


if __name__ == "__main__":
    unittest.main()
