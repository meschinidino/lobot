"""Unit tests: first-divergence index + highlight fallback (T8).

Doc: the UI marks the first token index where steered output differs from
baseline; token ids align to char spans; divergence at token 0 falls back
to a word-level diff. Pure functions, no model.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import app  # noqa: E402


class FakeTokenizer:
    """Token ids map 1:1 to words, so prefixes align exactly."""

    def __init__(self, words):
        self.words = words

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.words[i] for i in ids)


class FirstDivergenceTest(unittest.TestCase):
    def test_identical_is_none(self):
        self.assertIsNone(app.first_divergence([1, 2, 3], [1, 2, 3]))

    def test_both_empty_is_none(self):
        self.assertIsNone(app.first_divergence([], []))

    def test_divergence_at_zero(self):
        self.assertEqual(app.first_divergence([9, 2], [1, 2]), 0)

    def test_divergence_mid_stream(self):
        self.assertEqual(app.first_divergence([1, 2, 3], [1, 2, 4]), 2)

    def test_prefix_diverges_at_shorter_length(self):
        self.assertEqual(app.first_divergence([1, 2], [1, 2, 3]), 2)
        self.assertEqual(app.first_divergence([1, 2, 3], [1, 2]), 2)

    def test_empty_vs_nonempty_diverges_at_zero(self):
        self.assertEqual(app.first_divergence([], [1]), 0)


class HighlightTest(unittest.TestCase):
    WORDS = ["alpha", "beta", "gamma", "delta"]

    def test_identical_outputs_say_so(self):
        html = app.highlight_steered(FakeTokenizer(self.WORDS), "same text",
                                     [0, 1], None)
        self.assertIn("identical", html.lower())
        self.assertNotIn("<mark", html)

    def test_divergence_marks_suffix_only(self):
        tok = FakeTokenizer(self.WORDS)
        html = app.highlight_steered(tok, "alpha beta gamma delta",
                                     [0, 1, 2, 3], 2)
        self.assertIn("<mark", html)
        self.assertIn("alpha beta", html)
        # The marked region starts at the divergence cut.
        self.assertIn("<mark style='background:#ffe08a'> gamma delta</mark>",
                      html)

    def test_divergence_at_zero_defers_to_word_diff(self):
        tok = FakeTokenizer(self.WORDS)
        self.assertIsNone(
            app.highlight_steered(tok, "alpha beta", [0, 1], 0))

    def test_unaligned_prefix_defers_to_word_diff(self):
        class _BadTok:
            def decode(self, ids, skip_special_tokens=True):
                return "zzz-no-match"

        self.assertIsNone(
            app.highlight_steered(_BadTok(), "alpha beta", [0, 1], 1))

    def test_html_is_escaped(self):
        tok = FakeTokenizer(["<b>", "ok"])
        html = app.highlight_steered(tok, "<b> ok", [0, 1], None)
        self.assertIn("&lt;b&gt;", html)
        self.assertNotIn("<b> ok", html)


class WordDiffTest(unittest.TestCase):
    def test_changed_words_marked_equal_words_plain(self):
        html = app.highlight_word_diff("the cat sat", "the dog sat")
        self.assertIn("<mark", html)
        self.assertIn("dog", html)
        self.assertIn("the", html.split("<mark")[0])

    def test_identical_text_has_no_marks(self):
        html = app.highlight_word_diff("same words", "same words")
        self.assertNotIn("<mark", html)

    def test_empty_inputs_do_not_crash(self):
        html = app.highlight_word_diff("", "")
        self.assertIn("<div", html)


if __name__ == "__main__":
    unittest.main()
