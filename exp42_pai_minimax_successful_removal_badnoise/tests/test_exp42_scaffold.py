import unittest

from exp42_pai_minimax_successful_removal_badnoise.code.status import current_status


class Exp42ScaffoldTest(unittest.TestCase):
    def test_current_status(self) -> None:
        self.assertEqual(current_status(), "MINIMAX_SUCCESSFUL_REMOVAL_POOL_WEAK")
