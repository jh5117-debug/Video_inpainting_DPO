import unittest

import torch

from exp26_videopainter_dpo_v2.code.train_videopainter_dpo_adapter import (
    enforce_first_frame_gt_consistency,
    maybe_enforce_first_frame_gt,
)


class TestVP2FirstFrameConsistency(unittest.TestCase):
    def test_loser_and_condition_match_winner_first_frame(self):
        winner = torch.ones(3, 3, 4, 4)
        loser = torch.zeros_like(winner)
        conditioning = torch.zeros_like(winner)
        mask = torch.ones(3, 1, 4, 4)
        _, loser2, cond2, mask2 = enforce_first_frame_gt_consistency(winner, loser, conditioning, mask)
        self.assertTrue(torch.equal(loser2[0], winner[0]))
        self.assertTrue(torch.equal(cond2[0], winner[0]))
        self.assertTrue(torch.equal(mask2[0], torch.zeros_like(mask2[0])))

    def test_no_first_frame_gt_leaves_loser_unchanged(self):
        winner = torch.ones(3, 3, 4, 4)
        loser = torch.zeros_like(winner)
        conditioning = torch.zeros_like(winner)
        mask = torch.ones(3, 1, 4, 4)
        _, loser2, cond2, mask2 = maybe_enforce_first_frame_gt(winner, loser, conditioning, mask, enabled=False)
        self.assertTrue(torch.equal(loser2, loser))
        self.assertTrue(torch.equal(cond2, conditioning))
        self.assertTrue(torch.equal(mask2, mask))


if __name__ == "__main__":
    unittest.main()
