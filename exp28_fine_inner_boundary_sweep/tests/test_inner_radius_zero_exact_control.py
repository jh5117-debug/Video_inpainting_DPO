import unittest

import torch

from exp28_fine_inner_boundary_sweep.code.inner_boundary_geometry import (
    InnerBoundaryConfig,
    build_inner_boundary_regions,
    build_inner_boundary_weight_map,
    legacy_outer_one_ring,
)


class InnerRadiusZeroExactControlTest(unittest.TestCase):
    def test_radius_zero_matches_fresh_exp11_outer_control(self):
        image_hole = torch.zeros(1, 1, 1, 16, 16)
        image_hole[..., 4:12, 4:12] = 1.0
        loss_hole = torch.zeros(1, 1, 1, 8, 8)
        loss_hole[..., 2:6, 2:6] = 1.0

        cfg = InnerBoundaryConfig(inner_radius_px=0, inner_weight=0.75, outer_weight=0.75)
        regions = build_inner_boundary_regions(image_hole, loss_hole, cfg)
        expected_outer = legacy_outer_one_ring(loss_hole)
        expected_far = 1.0 - loss_hole - expected_outer
        expected_weight = loss_hole + 0.75 * expected_outer + 0.05 * expected_far
        weight = build_inner_boundary_weight_map(image_hole, loss_hole, cfg)

        self.assertTrue(torch.equal(regions["mask_core"], loss_hole))
        self.assertEqual(float(regions["inner_ring"].sum()), 0.0)
        self.assertTrue(torch.equal(regions["outer_ring"], expected_outer))
        self.assertLess(float((weight - expected_weight).abs().max()), 1e-7)


if __name__ == "__main__":
    unittest.main()
