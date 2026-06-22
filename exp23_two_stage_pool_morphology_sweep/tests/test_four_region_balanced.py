import unittest

import torch

from exp23_two_stage_pool_morphology_sweep.code.pool_morphology import PoolMorphologyConfig, build_pool_regions
from exp23_two_stage_pool_morphology_sweep.code.region_aggregation import effective_alpha, region_balanced_mean


class FourRegionBalancedTest(unittest.TestCase):
    def test_region_balanced_keeps_core_inner_outer_outside_independent(self):
        mask = torch.zeros(1, 1, 9, 9)
        mask[..., 3:6, 3:6] = 1.0
        regions = build_pool_regions(mask, PoolMorphologyConfig(pool_grid_scale=1, inner_pool_steps=1, outer_pool_steps=2))
        self.assertEqual(set(regions), {"mask_core", "inner_ring", "outer_ring", "far_outside"})
        alphas = {"mask_core": 1.0, "inner_ring": 0.25, "outer_ring": 0.75, "far_outside": 0.05}
        eff = effective_alpha(regions, alphas)
        self.assertEqual(eff["mask_core"], 1.0)
        self.assertEqual(eff["inner_ring"], 0.25)
        self.assertEqual(eff["outer_ring"], 0.75)
        self.assertEqual(eff["far_outside"], 0.05)
        out = region_balanced_mean(torch.ones_like(mask), regions, alphas)
        self.assertTrue(torch.allclose(out, torch.ones_like(out)))


if __name__ == "__main__":
    unittest.main()
