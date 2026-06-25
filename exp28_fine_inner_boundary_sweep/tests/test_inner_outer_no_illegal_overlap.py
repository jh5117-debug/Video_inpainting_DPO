import unittest

import torch

from exp28_fine_inner_boundary_sweep.code.inner_boundary_geometry import (
    InnerBoundaryConfig,
    build_inner_boundary_regions,
)


class InnerOuterNoIllegalOverlapTest(unittest.TestCase):
    def test_inner_and_outer_are_disjoint_on_loss_grid(self):
        image_hole = torch.zeros(1, 1, 1, 24, 24)
        image_hole[..., 4:20, 4:20] = 1.0
        loss_hole = torch.zeros(1, 1, 1, 6, 6)
        loss_hole[..., 1:5, 1:5] = 1.0

        regions = build_inner_boundary_regions(
            image_hole,
            loss_hole,
            InnerBoundaryConfig(inner_radius_px=4),
        )
        self.assertEqual(float((regions["inner_ring"] * regions["outer_ring"]).sum()), 0.0)
        self.assertLessEqual(float((regions["mask_core"] + regions["inner_ring"] - loss_hole).abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
