"""Unit tests: trait families + store (D12).

Proves: seed families carry template + rubric and start unvalidated;
resolve matches names case-free; unknown descriptions raise with
"not supported yet"; validation records persist across reload.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import trait_store  # noqa: E402
from trait_store import TraitFamily, TraitStore, UnsupportedTrait  # noqa: E402


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "traits.json"

    def test_seed_has_seven_unvalidated(self):
        store = TraitStore(path=self.path)
        fams = store.all()
        self.assertEqual(len(fams), 7)
        self.assertEqual(store.validated(), [])
        blunt = store.get("blunt")
        self.assertTrue(blunt.pos)
        self.assertTrue(blunt.judge)
        self.assertFalse(blunt.hand_validated)

    def test_resolve_matches_case_free(self):
        store = TraitStore(path=self.path)
        self.assertEqual(store.resolve("Blunt").name, "blunt")
        self.assertEqual(store.resolve("  PIRATE ").name, "pirate")

    def test_resolve_matches_alias(self):
        store = TraitStore(path=self.path, seed=False)
        store.register(TraitFamily(name="blunt", aliases=("direct",)))
        self.assertEqual(store.resolve("DIRECT").name, "blunt")

    def test_resolve_unknown_raises_not_supported(self):
        store = TraitStore(path=self.path)
        with self.assertRaises(UnsupportedTrait) as ctx:
            store.resolve("speak like a victorian ghost")
        self.assertIn("not supported yet", str(ctx.exception))

    def test_resolve_unknown_lists_validated(self):
        store = TraitStore(path=self.path)
        store.mark_validated("blunt", n_labels=12, agreement=0.95)
        with self.assertRaises(UnsupportedTrait) as ctx:
            store.resolve("speak like a victorian ghost")
        self.assertIn("blunt", str(ctx.exception))

    def test_register_duplicate_raises_unless_overwrite(self):
        store = TraitStore(path=self.path)
        with self.assertRaises(ValueError):
            store.register(TraitFamily(name="blunt"))
        store.register(TraitFamily(name="blunt", pos="new"),
                       overwrite=True)
        self.assertEqual(store.get("blunt").pos, "new")

    def test_mark_validated_records_gate(self):
        store = TraitStore(path=self.path)
        fam = store.mark_validated("blunt", n_labels=12, agreement=0.95,
                                   note="O1 gate")
        self.assertTrue(fam.hand_validated)
        self.assertEqual(fam.validation["hand_labels"], 12)
        self.assertEqual(fam.validation["agreement"], 0.95)
        self.assertEqual([f.name for f in store.validated()], ["blunt"])
        with self.assertRaises(KeyError):
            store.mark_validated("ghost", n_labels=1, agreement=1.0)

    def test_save_load_roundtrip(self):
        store = TraitStore(path=self.path)
        store.mark_validated("blunt", n_labels=12, agreement=0.95)
        store.save()
        again = TraitStore(path=self.path)
        blunt = again.get("blunt")
        self.assertTrue(blunt.hand_validated)
        self.assertEqual(blunt.validation["agreement"], 0.95)
        self.assertEqual(len(again.all()), 7)

    def test_empty_store_without_seed(self):
        store = TraitStore(path=self.path, seed=False)
        self.assertEqual(store.all(), [])


if __name__ == "__main__":
    unittest.main()
