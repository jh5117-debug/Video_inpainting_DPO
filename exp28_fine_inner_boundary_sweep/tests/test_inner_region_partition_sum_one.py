import unittest

import torch

from exp28_fine_inner_boundary_sweep.code.inner_boundary_geometry import (
    InnerBoundaryConfig,
    build_inner_boundary_regions,
)


class InnerRegionPartitionSumOneTest(unittest.TestCase):
    def test_regions_are_nonnegative_and_sum_to_one(self):
        image_hole = torch.zeros(1, 1, 1, 32, 32)
        image_hole[..., 8:25, 6:27] = 1.0
        loss_hole = torch.zeros(1, 1, 1, 8, 8)
        loss_hole[..., 2:7, 2:7] = 1.0

        for radius in (0, 2, 4, 8):
            regions = build_inner_boundary_regions(
                image_hole,
                loss_hole,
                InnerBoundaryConfig(inner_radius_px=radius),
            )
            total = sum(regions.values())
            self.assertLess(float((total - 1.0).abs().max()), 1e-6)
            for region in regions.values():
                self.assertGreaterEqual(float(region.min()), -1e-6)
                self.assertLessEqual(float(region.max()), 1.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
