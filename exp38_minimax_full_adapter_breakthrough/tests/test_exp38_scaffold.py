import unittest

from exp38_minimax_full_adapter_breakthrough.code import status


class Exp38ScaffoldTest(unittest.TestCase):
    def test_status_constants(self) -> None:
        self.assertEqual(
            status.EXPERIMENT_ID,
            "exp38_minimax_full_adapter_breakthrough",
        )
        self.assertEqual(status.READBACK_STATUS, "EXP38_READBACK_COMPLETED")
        self.assertIn("exp38-minimax-full-adapter", status.BRANCH)


if __name__ == "__main__":
    unittest.main()

