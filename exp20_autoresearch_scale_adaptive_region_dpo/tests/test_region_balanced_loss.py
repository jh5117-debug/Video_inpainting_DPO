import unittest

import torch

from exp20_autoresearch_scale_adaptive_region_dpo.code.region_balanced_loss import compute_dpo_loss


def toy_regions():
    mask = torch.zeros(2, 1, 4, 4)
    mask[:, :, 1:3, 1:3] = 1
    boundary = torch.zeros_like(mask)
    boundary[:, :, 0, 1:3] = 1
    boundary[:, :, 3, 1:3] = 1
    outside = torch.clamp(1 - mask - boundary, 0, 1)
    return {"mask": mask, "boundary": boundary, "outside": outside}


class RegionBalancedLossTests(unittest.TestCase):
    def test_legacy_loss_finite_and_grad(self):
        regions = toy_regions()
        err_w = (torch.ones(2, 1, 4, 4) * 0.8).requires_grad_()
        err_l = (torch.ones(2, 1, 4, 4) * 1.4).requires_grad_()
        err_w_ref = torch.ones(2, 1, 4, 4) * 1.0
        err_l_ref = torch.ones(2, 1, 4, 4) * 1.0
        loss, diag = compute_dpo_loss(err_w, err_l, err_w_ref, err_l_ref, regions)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(err_w.grad)
        self.assertIn("dpo_loss", diag)

    def test_region_balanced_empty_boundary(self):
        regions = toy_regions()
        regions["boundary"] = torch.zeros_like(regions["boundary"])
        err_w = (torch.ones(2, 1, 4, 4) * 0.8).requires_grad_()
        err_l = (torch.ones(2, 1, 4, 4) * 1.2).requires_grad_()
        err_w_ref = torch.ones(2, 1, 4, 4)
        err_l_ref = torch.ones(2, 1, 4, 4)
        loss, diag = compute_dpo_loss(
            err_w,
            err_l,
            err_w_ref,
            err_l_ref,
            regions,
            aggregation="region_balanced",
            boundary_weight=0.75,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(err_l.grad)
        self.assertEqual(diag["aggregation"], "region_balanced")


if __name__ == "__main__":
    unittest.main()
