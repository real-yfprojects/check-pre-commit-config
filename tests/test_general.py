"""Test small stuff that doesn't belong in any of the other files."""
import unittest

from check_pre_commit_config_frozen import process_frozen_comment, strip_rich_markup


class GeneralTests(unittest.TestCase):
    """General independent tests."""

    def test_strip_rich(self):
        """Test the `strip_rich_markup` function."""
        strings = [
            ("Some string with abitrary letters, kommas, ... Exclamation!",) * 2,
            (
                "Some string with [green]markup and more.[/green]",
                "Some string with markup and more.",
            ),
            ("This doesn't \\[contain] markup",) * 2,
            ("This doesn't \\[contain] markup \\[either\\]",) * 2,
        ]

        for s, e in strings:
            with self.subTest(s=s):
                self.assertEqual(strip_rich_markup(s), e)

    def test_comment(self):
        """Test the `process_frozen_comment` function."""
        frozen_comments = [
            ("# frozen: rev", "rev", None),
            ("# frozen: rev ", "rev", " "),
            ("# frozen: rev some comment", "rev", " some comment"),
            ("# frozen: rev # some comment", "rev", " # some comment"),
        ]
        unfrozen_comments = [
            "#frozen: rev",
            "# frozen: ",
            "# frozen",
            "# frozen:",
            "# frozen: ",
            "#",
            "# Some normal comment ",
            "#Some normal comment ",
            "Some comment",
            "Some # comment",
            " Some comment",
            "frozen: rev",
            " frozen: rev",
            " frozen: rev # some comment",
            "frozen: rev # some comment",
        ]

        for c, expected_rev, expected_note in frozen_comments:
            with self.subTest(comment=c):
                rev, note = process_frozen_comment(c)
                self.assertEqual(rev, expected_rev)
                self.assertEqual(note, expected_note)

        for c in unfrozen_comments:
            with self.subTest("Invalid", comment=c):
                rev, note = process_frozen_comment(c)
                self.assertIsNone(rev)
                self.assertEqual(note, c)
