import unittest

import torch

from exp28_fine_inner_boundary_sweep.code.inner_boundary_geometry import (
    InnerBoundaryConfig,
    build_inner_boundary_regions,
    legacy_outer_one_ring,
)


class OuterRingLegacyExactUnchangedTest(unittest.TestCase):
    def test_outer_ring_is_identical_for_all_main_inner_radii(self):
        image_hole = torch.zeros(1, 2, 1, 24, 24)
        image_hole[..., 5:17, 7:20] = 1.0
        loss_hole = torch.zeros(1, 2, 1, 6, 6)
        loss_hole[..., 2:5, 2:5] = 1.0
        expected_outer = legacy_outer_one_ring(loss_hole)

        for radius in (2, 4, 8):
            regions = build_inner_boundary_regions(
                image_hole,
                loss_hole,
                InnerBoundaryConfig(inner_radius_px=radius),
            )
            self.assertEqual(float((regions["outer_ring"] - expected_outer).abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
