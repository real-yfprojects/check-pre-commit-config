"""Test the linter."""
import unittest

from check_pre_commit_config_frozen import Complaint, Linter, Rule


class ComplainTest(unittest.TestCase):
    """Test whether all complaints are issued correctly."""

    pass  # TODO


class GeneralTest(unittest.TestCase):
    """Test other units of the linter."""

    def setUp(self) -> None:
        """Prepare some rules and corrensponding complaints."""
        self.rule1 = Rule.FORCE_FREEZE
        self.rule1.template = ""
        self.rule2 = Rule.FORCE_UNFREEZE
        self.rule2.template = ""
        self.complaint1 = Complaint("", 0, 0, self.rule1, "", True)
        self.complaint2 = Complaint("", 0, 0, self.rule2, "", True)

    def test_enabled(self):
        """Test the `enabled` method of `Linter`."""
        linter = Linter({self.rule1.code}, set())

        with self.subTest("Enabled Rule"):
            self.assertTrue(linter.enabled(self.rule1))
        with self.subTest("Disabled Rule"):
            self.assertFalse(linter.enabled(self.rule2))

        with self.subTest("Enabled Complain"):
            self.assertTrue(linter.enabled(self.complaint1))
        with self.subTest("Disabled Complain"):
            self.assertFalse(linter.enabled(self.complaint2))

    def test_should_fix(self):
        """Test the `should_fix` method of `Linter`."""
        linter = Linter(set(), {self.rule1.code})
        with self.subTest("Enabled, do not fix"):
            self.assertFalse(linter.should_fix(self.rule1))
            self.assertFalse(linter.should_fix(self.complaint1))
        with self.subTest("Disabled, do not fix"):
            self.assertFalse(linter.should_fix(self.rule2))
            self.assertFalse(linter.should_fix(self.complaint2))

        linter = Linter({self.rule1.code}, {self.rule1.code, self.rule2.code})
        with self.subTest("Enabled, fix"):
            self.assertTrue(linter.should_fix(self.rule1))
            self.assertTrue(linter.should_fix(self.complaint1))
        with self.subTest("Disabled, fix"):
            self.assertFalse(linter.should_fix(self.rule2))
            self.assertFalse(linter.should_fix(self.complaint2))

        self.complaint1 = Complaint("", 0, 0, self.rule1, "", False)

        with self.subTest("Enabled, fix, not fixable"):
            self.assertFalse(linter.should_fix(self.complaint1))

    def test_complain(self):
        """Test the `complain` method."""
        linter = Linter({self.rule1.code}, set())

        self.assertDictEqual(linter._complains, {})

        linter.complain("", self.rule2, 0, 0, False)

        self.assertDictEqual(linter._complains, {})

        linter.complain("", self.rule1, 0, 0, False)

        self.assertIn("", linter._complains)
        self.assertEqual(len(linter._complains[""]), 1)

        # test adding to existing complains
        linter.complain("", self.rule1, 0, 0, False)
        self.assertEqual(len(linter._complains[""]), 2)
