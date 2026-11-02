"""No MOD compatibility shims: old names and re-export modules are gone."""
import importlib.util
import unittest

from model_lobotomy import guards, trait_eval


class NoShimsTest(unittest.TestCase):
    def test_job_import_module_gone(self):
        self.assertIsNone(importlib.util.find_spec("model_lobotomy.job_import"))

    def test_lane_old_judge_name_gone(self):
        self.assertTrue(hasattr(trait_eval.TorchLane, "score_many"))
        self.assertFalse(hasattr(trait_eval.TorchLane, "judge_many"))

    def test_guard_wrapper_old_surface_gone(self):
        for name in ("check", "check_verified", "check_confirmed",
                     "check_full", "check_screen"):
            self.assertFalse(hasattr(guards.ScreensV2Guard, name), name)
            self.assertFalse(hasattr(guards.NoopGuard, name), name)

    def test_no_placeholder_words_in_src(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        bad = []
        for base in ("src", "web/js"):
            for p in (root / base).rglob("*"):
                if p.is_file() and p.suffix in (".py", ".js"):
                    text = p.read_text(errors="ignore")
                    for word in ("migrate at their own pace", "pre-MOD",
                                 "no-op stage"):
                        if word in text:
                            bad.append(f"{p}: {word}")
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
