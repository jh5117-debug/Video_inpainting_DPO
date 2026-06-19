import unittest

import torch

from exp20_autoresearch_scale_adaptive_region_dpo.code.region_balanced_loss import exp11_dpo_from_losses


class Exp11DiagnosticParityTests(unittest.TestCase):
    def test_loser_degrade_ratio_uses_pair_gap_definition(self):
        m_w = torch.tensor([0.80, 0.80, 1.20, 1.20])
        m_w_ref = torch.ones(4)
        m_l = torch.tensor([1.40, 1.40, 1.10, 1.10])
        m_l_ref = torch.ones(4)
        _, diag = exp11_dpo_from_losses(m_w, m_l, m_w_ref, m_l_ref, nframes=2)

        pair_win = torch.log((m_w + 1e-8) / (m_w_ref + 1e-8)).view(-1, 2).mean(dim=1)
        pair_lose = torch.log((m_l + 1e-8) / (m_l_ref + 1e-8)).view(-1, 2).mean(dim=1).clamp(max=1.0)
        inside = -0.5 * 10.0 * (pair_win - 0.25 * pair_lose)
        correct = inside > 0
        winner_improvement = (-pair_win).clamp(min=0)
        loser_degradation = pair_lose.clamp(min=0)
        expected_count = ((correct) & (loser_degradation > winner_improvement)).sum().float()
        expected_ratio = expected_count / correct.sum().float().clamp(min=1)

        self.assertAlmostEqual(float(diag["loser_degrade_count"]), float(expected_count))
        self.assertAlmostEqual(float(diag["loser_degrade_ratio"]), float(expected_ratio))
        self.assertTrue(torch.equal(diag["_inside_term"], inside))
        self.assertTrue(torch.equal(diag["_winner_improvement"], winner_improvement))
        self.assertTrue(torch.equal(diag["_loser_degradation"], loser_degradation))


if __name__ == "__main__":
    unittest.main()
