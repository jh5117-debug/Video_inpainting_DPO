import unittest

from exp42_pai_minimax_successful_removal_badnoise.code.status import current_status


class Exp42ScaffoldTest(unittest.TestCase):
    def test_current_status_readback(self) -> None:
        self.assertEqual(current_status(), "EXP42_PAI_MINIMAX_DATA_READBACK_COMPLETED")
