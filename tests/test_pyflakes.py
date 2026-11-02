"""Pyflakes gate: no undefined names and no unused imports in src/scripts."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class PyflakesTest(unittest.TestCase):
    def test_pyflakes_clean(self):
        proc = subprocess.run(
            [sys.executable, "-m", "pyflakes", "src", "scripts"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
