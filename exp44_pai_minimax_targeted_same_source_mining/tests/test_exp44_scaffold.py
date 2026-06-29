import unittest

from exp44_pai_minimax_targeted_same_source_mining.code.status import current_status


class Exp44ScaffoldTest(unittest.TestCase):
    def test_current_status(self) -> None:
        self.assertEqual(current_status(), "EXP44_TARGETED_READBACK_COMPLETED")


if __name__ == "__main__":
    unittest.main()
