import unittest

import torch

from exp26_videopainter_dpo_v2.code.train_videopainter_dpo_adapter import formal_loser_dominant_stats


class TestVP2LoserDominantDefinition(unittest.TestCase):
    def test_uses_correct_preference_and_degradation_vs_improvement(self):
        inside = torch.tensor([1.0, 1.0, -1.0])
        win_gap = torch.tensor([-0.1, -0.5, -0.1])
        lose_gap = torch.tensor([0.2, 0.1, 10.0])
        stats = formal_loser_dominant_stats(inside, win_gap, lose_gap)
        # Only first correct pair is loser-dominant; third is not correct.
        self.assertAlmostEqual(float(stats["loser_dominant_ratio"]), 0.5)
        self.assertEqual(float(stats["loser_degrade_count"]), 1.0)
        self.assertEqual(float(stats["n_correct"]), 2.0)
        self.assertEqual(float(stats["n_total"]), 3.0)


if __name__ == "__main__":
    unittest.main()
