import math
import unittest

import torch

from exp57_void_adaptive_transition_safe.adaptive_transition_loss import (
    AdaptiveLossConfig,
    build_adaptive_loss,
    config_for_cell,
    grad_stats,
    select_backtracking_scale,
    transition_risk_weights,
)


class AdaptiveTransitionLossTests(unittest.TestCase):
    def test_loser_lambda_zero_when_conflicting(self):
        winner = torch.tensor([1.0, 0.0])
        loser = torch.tensor([-1.0, 0.0])
        stats = grad_stats(winner, loser, 0.25)
        self.assertEqual(stats["lambda_loser_global"], 0.0)
        self.assertLess(stats["gradient_dot"], 0.0)

    def test_loser_lambda_positive_when_aligned(self):
        winner = torch.tensor([1.0, 0.0])
        loser = torch.tensor([0.5, 0.0])
        stats = grad_stats(winner, loser, 0.25)
        self.assertGreater(stats["lambda_loser_global"], 0.0)
        self.assertLessEqual(stats["lambda_loser_global"], 0.25)

    def test_transition_risk_increases_preservation_and_downscales_object(self):
        cfg = AdaptiveLossConfig()
        weights = transition_risk_weights({"overlap": 3e-5, "affected": 0.0, "boundary": 3e-5, "outside": 0.0}, cfg)
        self.assertFalse(weights["transition_safe_pass"])
        self.assertLess(weights["object_dpo_scale_final"], cfg.object_dpo_base)
        self.assertGreater(weights["overlap_pres_weight_final"], cfg.overlap_pres_base)
        self.assertGreater(weights["boundary_pres_weight_final"], cfg.boundary_pres_base)

    def test_backtracking_reduces_scale(self):
        cfg = AdaptiveLossConfig()
        decision = select_backtracking_scale(
            [
                {"scale": 1.0, "winner": 0.0, "overlap": 3e-5, "affected": 0.0, "boundary": 0.0, "outside": 0.0},
                {"scale": 0.5, "winner": 0.0, "overlap": 1e-5, "affected": 0.0, "boundary": 0.0, "outside": 0.0},
            ],
            cfg,
        )
        self.assertFalse(decision["update_rejected"])
        self.assertEqual(decision["finite_diff_selected_scale"], 0.5)

    def test_backtracking_rejects_when_no_scale_safe(self):
        cfg = AdaptiveLossConfig()
        decision = select_backtracking_scale(
            [
                {"scale": 1.0, "winner": 0.0, "overlap": 3e-5, "affected": 0.0, "boundary": 0.0, "outside": 0.0},
                {"scale": 0.5, "winner": 0.0, "overlap": 3e-5, "affected": 0.0, "boundary": 0.0, "outside": 0.0},
            ],
            cfg,
        )
        self.assertTrue(decision["update_rejected"])
        self.assertEqual(decision["finite_diff_selected_scale"], 0.0)

    def test_zero_gap_loss_near_log_two(self):
        cfg = AdaptiveLossConfig(winner_anchor_base=0.0, object_pres_base=0.0, overlap_pres_base=0.0, affected_pres_base=0.0, boundary_pres_base=0.0, outside_pres_base=0.0)
        p = {r: torch.tensor(1.0) for r in ("object", "overlap", "affected", "boundary", "outside")}
        ref = {r: torch.tensor(1.0) for r in ("object", "overlap", "affected", "boundary", "outside")}
        loss, info = build_adaptive_loss(p, ref, p, ref, cfg, {"lambda_loser_global": 0.0})
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss), math.log(2.0), places=5)
        self.assertAlmostEqual(float(info["preference_margin"]), 0.0, places=7)

    def test_outside_has_no_loser_dpo_by_default(self):
        cfg = config_for_cell("ATS0_Q2_T500_S0")
        self.assertEqual(cfg.dpo_scales["outside"], 0.0)


if __name__ == "__main__":
    unittest.main()
