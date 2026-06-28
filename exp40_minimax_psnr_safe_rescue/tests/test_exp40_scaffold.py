import unittest

from exp40_minimax_psnr_safe_rescue.code import status


class Exp40ScaffoldTest(unittest.TestCase):
    def test_status_and_forbidden_claims(self):
        self.assertEqual(status.current_status(), "EXP40_READBACK_COMPLETED")
        self.assertIn("UNIVERSAL_ADAPTER", status.FORBIDDEN_CLAIMS)


if __name__ == "__main__":
    unittest.main()
