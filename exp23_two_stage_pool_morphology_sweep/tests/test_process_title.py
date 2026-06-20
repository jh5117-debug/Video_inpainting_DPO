import unittest

from exp23_two_stage_pool_morphology_sweep.code.process_title import set_process_title


class ProcessTitleTest(unittest.TestCase):
    def test_process_title_best_effort(self):
        status = set_process_title("Phy")
        self.assertEqual(status["requested"], "Phy")
        self.assertIn("prctl", status)


if __name__ == "__main__":
    unittest.main()
