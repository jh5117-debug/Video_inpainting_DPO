import unittest

from exp42_pai_minimax_successful_removal_badnoise.scripts.mine_successful_removal_candidates import auto_classify


class Exp42MiningClassificationTest(unittest.TestCase):
    def test_successful_candidate_threshold(self) -> None:
        label, _ = auto_classify(
            {
                "full_psnr": 28.0,
                "mask_psnr": 24.0,
                "boundary_psnr": 24.0,
                "outside_psnr": 29.0,
                "outside_mae": 3.0,
                "temporal_diff_mae": 5.0,
            }
        )
        self.assertEqual(label, "SUCCESSFUL_REMOVAL_CANDIDATE")

    def test_trivial_bad_outside_failure(self) -> None:
        label, _ = auto_classify(
            {
                "full_psnr": 23.0,
                "mask_psnr": 21.0,
                "boundary_psnr": 21.0,
                "outside_psnr": 19.0,
                "outside_mae": 19.0,
                "temporal_diff_mae": 4.0,
            }
        )
        self.assertEqual(label, "TRIVIAL_BAD")
