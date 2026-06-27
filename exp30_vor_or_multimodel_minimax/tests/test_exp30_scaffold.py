import unittest

from exp30_vor_or_multimodel_minimax.code.exp30_status import (
    EXP30_BRANCH,
    FORBIDDEN_CLAIMS,
    current_scope,
)


class Exp30ScaffoldTest(unittest.TestCase):
    def test_scope_mentions_minimax(self):
        self.assertIn("MiniMax", current_scope())

    def test_forbidden_universal_claim(self):
        self.assertIn("UNIVERSAL_ADAPTER", FORBIDDEN_CLAIMS)

    def test_branch_name(self):
        self.assertTrue(EXP30_BRANCH.startswith("research/exp30-"))


if __name__ == "__main__":
    unittest.main()

