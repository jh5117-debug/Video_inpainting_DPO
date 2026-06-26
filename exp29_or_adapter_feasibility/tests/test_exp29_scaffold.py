import unittest

from exp29_or_adapter_feasibility.code.exp29_status import (
    is_known_effecterase_state,
    is_known_minimax_state,
    is_known_summary_role,
)


class Exp29ScaffoldTest(unittest.TestCase):
    def test_final_state_enums(self):
        self.assertTrue(is_known_minimax_state("MINIMAX_BLOCKED"))
        self.assertTrue(is_known_effecterase_state("EFFECTERASE_BLOCKED"))
        self.assertFalse(is_known_minimax_state("UNIVERSAL_ADAPTER"))

    def test_summary_roles(self):
        self.assertTrue(is_known_summary_role("OR_STRONG_BASELINE"))
        self.assertFalse(is_known_summary_role("FINAL_SOTA"))


if __name__ == "__main__":
    unittest.main()

