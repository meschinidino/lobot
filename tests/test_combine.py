"""Unit tests: combined patches (CB).

No model. Proves: edits_for sums same-layer vectors order-free
and merges ablations; overlap warnings fire on collinear
vector/ablation pairs; the matrix carries singles, pairs, and
the full set with cache flags; install order is safest-first;
routes return 400/404/409/422 in the right cases.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import combine as C  # noqa: E402
from model_lobotomy import guard  # noqa: E402
from model_lobotomy import installers as I  # noqa: E402
from model_lobotomy import patch as patch_mod  # noqa: E402
from model_lobotomy import server  # noqa: E402
from tests.test_traits_api import _write_job  # noqa: E402


def _patch(direction, layer=7, dose=1.0, scale=2.0, trait="x",
           **meta_kw):
    meta = {"layer": layer, "dose": dose, "scale": scale,
            "trait": trait}
    meta.update(meta_kw)
    return {"direction": list(direction), "meta": meta}


def _verdict(blocked):
    return guard.GuardVerdict(
        blocked=blocked, min_ratio=0.1 if blocked else 1.0,
        worst_prompt=None, blocked_prompts=(), natural=(),
        steered=(), cached=False, research_mode=False,
        would_block=blocked, full_generation=True, screens=())


class FakeGuard:
    def __init__(self, block_when):
        self.block_when = block_when
        self.calls: list = []

    def check_verified(self, engine, edits, *, medians,
                       max_new_tokens, seed, judge_fn,
                       cancel=None):
        self.calls.append(edits)
        return _verdict(self.block_when(edits))


class CachingFakeGuard(FakeGuard):
    def __init__(self, block_when):
        super().__init__(block_when)
        self.cache: dict = {}

    def check_verified(self, engine, edits, *, medians,
                       max_new_tokens, seed, judge_fn,
                       cancel=None):
        key = guard.edit_set_fingerprint(edits)
        if key in self.cache:
            v = self.cache[key]
            return guard.GuardVerdict(
                blocked=v.blocked, min_ratio=v.min_ratio,
                worst_prompt=v.worst_prompt,
                blocked_prompts=v.blocked_prompts,
                natural=v.natural, steered=v.steered, cached=True,
                research_mode=False, would_block=v.would_block,
                full_generation=True, screens=())
        v = super().check_verified(
            engine, edits, medians=medians,
            max_new_tokens=max_new_tokens, seed=seed,
            judge_fn=judge_fn, cancel=cancel)
        self.cache[key] = v
        return v


class EditsForTest(unittest.TestCase):
    def test_sums_same_layer_vectors_order_free(self):
        a = _patch([1.0, 0.0], layer=7, dose=1.0)
        b = _patch([0.0, 1.0], layer=7, dose=2.0)
        e1 = C.edits_for([a, b])
        e2 = C.edits_for([b, a])
        self.assertEqual(e1, e2)
        self.assertEqual(len(e1.vector), 2)
        self.assertEqual([v.dose for v in e1.vector], [1.0, 2.0])

    def test_stacks_different_layers(self):
        a = _patch([1.0], layer=9)
        b = _patch([2.0], layer=7)
        e = C.edits_for([a, b])
        self.assertEqual([v.layer for v in e.vector], [7, 9])

    def test_merges_ablations(self):
        a = _patch([1.0, 0.0], layer=7, edit_kind="ablate")
        b = _patch([0.0, 1.0], layer=5, edit_kind="ablate")
        v = _patch([1.0, 1.0], layer=7)
        e = C.edits_for([a, b, v])
        self.assertEqual([x.layer for x in e.ablate], [5, 7])
        self.assertEqual(len(e.vector), 1)

    def test_neuron_rows_from_meta(self):
        a = _patch([1.0], layer=7,
                   rows=[{"layer": 3, "neuron": 9, "dose": 0.5}])
        e = C.edits_for([a])
        self.assertEqual(len(e.neuron), 1)
        self.assertEqual(e.neuron[0].neuron, 9)

    def test_tensor_direction(self):
        a = {"direction": torch.tensor([1.0, 2.0]),
             "meta": {"layer": 7, "dose": 1.0, "scale": 2.0,
                      "trait": "x"}}
        e = C.edits_for([a])
        self.assertEqual(tuple(e.vector[0].direction), (1.0, 2.0))


class OverlapTest(unittest.TestCase):
    def test_collinear_vector_ablation_warns(self):
        v = _patch([1.0, 0.0], layer=7)
        a = _patch([0.9, 0.1], layer=7, edit_kind="ablate")
        w = C.overlap_warnings(C.edits_for([v, a]))
        self.assertEqual(len(w), 1)
        self.assertIn("L7", w[0])

    def test_orthogonal_is_quiet(self):
        v = _patch([1.0, 0.0], layer=7)
        a = _patch([0.0, 1.0], layer=7, edit_kind="ablate")
        self.assertEqual(C.overlap_warnings(C.edits_for([v, a])),
                         [])

    def test_other_layer_is_quiet(self):
        v = _patch([1.0, 0.0], layer=7)
        a = _patch([1.0, 0.0], layer=8, edit_kind="ablate")
        self.assertEqual(C.overlap_warnings(C.edits_for([v, a])),
                         [])

    def test_neuron_overlap_joins(self):
        a = _patch([1.0, 0.0], layer=7, edit_kind="ablate")
        n = _patch([0.0], layer=7,
                   rows=[{"layer": 7, "neuron": 0, "dose": 1.0}])
        w = C.overlap_warnings(C.edits_for([a, n]))
        self.assertEqual(len(w), 1)
        self.assertIn("L7N0", w[0])


class MatrixTest(unittest.TestCase):
    def _run(self, guard):
        patches = [_patch([1.0, 0.0], layer=7, trait="a"),
                   _patch([0.0, 1.0], layer=8, trait="b")]
        return C.pair_matrix(
            ["a", "b"], patches, engine=None, medians={},
            max_new_tokens=8, seed=0,
            judge_fn=lambda t: 0.0, guard=guard)

    def test_shape_and_counts(self):
        g = FakeGuard(lambda edits: False)
        out = self._run(g)
        self.assertEqual(len(out["singles"]), 2)
        self.assertEqual(len(out["pairs"]), 1)
        self.assertEqual(out["pairs"][0]["patches"], ["a", "b"])
        self.assertEqual(out["full_set"]["patches"], ["a", "b"])
        self.assertEqual(out["checks"], 4)
        self.assertEqual(out["cached"], 0)
        self.assertIn("seconds", out)

    def test_full_set_blocks_when_pair_blocks(self):
        g = FakeGuard(lambda edits: len(edits.vector) >= 2)
        out = self._run(g)
        self.assertFalse(any(s["blocked"] for s in out["singles"]))
        self.assertTrue(out["pairs"][0]["blocked"])
        self.assertTrue(out["full_set"]["blocked"])

    def test_cache_hit_flag(self):
        g = CachingFakeGuard(lambda edits: False)
        out1 = self._run(g)
        # With two patches the full set repeats the pair union.
        self.assertEqual(out1["cached"], 1)
        out2 = self._run(g)
        self.assertEqual(out2["cached"], 4)

    def test_three_patches(self):
        patches = [_patch([1.0], layer=i, trait=t)
                   for i, t in enumerate("abc")]
        g = FakeGuard(lambda edits: False)
        out = C.pair_matrix(
            ["a", "b", "c"], patches, engine=None, medians={},
            max_new_tokens=8, seed=0, judge_fn=lambda t: 0.0,
            guard=g)
        self.assertEqual(len(out["pairs"]), 3)
        self.assertEqual(out["checks"], 3 + 3 + 1)


class InstallOrderTest(unittest.TestCase):
    def test_safest_first(self):
        blocked = {frozenset({"a"}): set(),
                   frozenset({"b"}): {"honesty"},
                   frozenset({"c"}): set(),
                   frozenset({"a", "b"}): {"honesty"},
                   frozenset({"a", "c"}): set(),
                   frozenset({"b", "c"}): {"honesty"},
                   frozenset({"a", "b", "c"}): {"honesty",
                                               "privacy"}}

        def fn(sub):
            return blocked[frozenset(sub)]

        self.assertEqual(C.install_order(["a", "b", "c"], fn),
                         ["a", "c", "b"])

    def test_ties_break_by_name(self):
        self.assertEqual(C.install_order(["b", "a"],
                                         lambda sub: set()),
                         ["a", "b"])


class RouteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.jobs_dir = Path(self.tmp.name) / "jobs"
        self.jobs_dir.mkdir()
        from model_lobotomy import create_job
        server._runner = create_job.CreateRunner(
            jobs_dir=self.jobs_dir)
        self.addCleanup(setattr, server, "_runner", None)
        self.roots = mock.patch.object(
            I, "INSTALLS_ROOT", Path(self.tmp.name) / "installs")
        self.roots.start()
        self.addCleanup(self.roots.stop)
        self.client = TestClient(server.api)
        self.job_a = "blunt-0-20260101-000000-000"
        self.job_b = "blunt-0-20260101-000001-001"
        for job in (self.job_a, self.job_b):
            job_dir = _write_job(self.jobs_dir, job, trait="blunt")
            patch_mod.save_patch(
                job_dir / "trait" / "knob.lobot",
                direction=torch.ones(8), layer=14, dose=0.5,
                trait="blunt", recipe={},
                report_card={}, guard={},
                fingerprint={
                    "model_id":
                    server.engine._active_model_id},
                scale=2.0)

    def test_cap_400_above_8(self):
        with mock.patch.object(server, "NO_MODEL", False):
            r = self.client.post("/api/combine/check", json={
                "patch_ids": [f"id-{i}" for i in range(9)]})
        self.assertEqual(r.status_code, 400)

    def test_empty_400(self):
        with mock.patch.object(server, "NO_MODEL", False):
            r = self.client.post("/api/combine/check", json={
                "patch_ids": []})
        self.assertEqual(r.status_code, 400)

    def test_unknown_404(self):
        with mock.patch.object(server, "NO_MODEL", False), \
                mock.patch.object(server, "get_engine"):
            r = self.client.post("/api/combine/check", json={
                "patch_ids": ["nope-0"]})
        self.assertEqual(r.status_code, 404)

    def test_missing_patch_422(self):
        (self.jobs_dir / self.job_a / "trait"
         / "knob.lobot").unlink()
        with mock.patch.object(server, "NO_MODEL", False), \
                mock.patch.object(server, "get_engine"):
            r = self.client.post("/api/combine/check", json={
                "patch_ids": [self.job_a]})
        self.assertEqual(r.status_code, 422)

    def test_model_mismatch_422(self):
        job_dir = self.jobs_dir / self.job_a
        patch_mod.save_patch(
            job_dir / "trait" / "knob.lobot",
            direction=torch.ones(8), layer=14, dose=0.5,
            trait="blunt", recipe={}, report_card={}, guard={},
            fingerprint={"model_id": "other/model"}, scale=2.0)
        with mock.patch.object(server, "NO_MODEL", False), \
                mock.patch.object(server, "get_engine"):
            r = self.client.post("/api/combine/check", json={
                "patch_ids": [self.job_a]})
        self.assertEqual(r.status_code, 422)

    def test_busy_409(self):
        eng = mock.Mock()
        eng.lock.locked.return_value = True
        with mock.patch.object(server, "NO_MODEL", False), \
                mock.patch.object(server, "get_engine",
                                  return_value=eng):
            r = self.client.post("/api/combine/check", json={
                "patch_ids": [self.job_a]})
        self.assertEqual(r.status_code, 409)

    def test_check_ok_and_dedupe(self):
        eng = mock.Mock()
        eng.lock.locked.return_value = False
        sent = {"matrix": {"install_order": ["blunt",
                                             "blunt #2"]}}
        with mock.patch.object(server, "NO_MODEL", False), \
                mock.patch.object(server, "get_engine",
                                  return_value=eng), \
                mock.patch.object(server, "_combine_matrix",
                                  return_value=dict(sent)) as cm:
            r = self.client.post("/api/combine/check", json={
                "patch_ids": [self.job_a, self.job_b]})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), sent)
        names = cm.call_args.args[0]
        self.assertEqual(names, ["blunt", "blunt #2"])

    def test_installed_empty_no_model(self):
        r = self.client.get("/api/combine/installed")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(),
                         {"installed": [], "matrix": None})

    def test_installed_runs_matrix(self):
        rec = {"install_id": "x", "trait_id": self.job_a,
               "target": "llamacpp"}
        d = I.INSTALLS_ROOT / "x"
        d.mkdir(parents=True)
        (d / "install.json").write_text(json.dumps(rec))
        eng = mock.Mock()
        eng.lock.locked.return_value = False
        sent = {"install_order": ["blunt"]}
        with mock.patch.object(server, "NO_MODEL", False), \
                mock.patch.object(server, "get_engine",
                                  return_value=eng), \
                mock.patch.object(server, "_combine_matrix",
                                  return_value=dict(sent)):
            r = self.client.get("/api/combine/installed")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["installed"], [self.job_a])
        self.assertEqual(r.json()["matrix"], sent)


if __name__ == "__main__":
    unittest.main()
