import unittest

import torch

from exp23_two_stage_pool_morphology_sweep.code.pool_morphology import (
    PoolMorphologyConfig,
    build_pool_regions,
    build_region_weight_map,
    dilate_steps,
    erode_steps,
)
from exp23_two_stage_pool_morphology_sweep.code.region_aggregation import (
    legacy_weighted_mean,
    region_balanced_mean,
)


class PoolMorphologyTest(unittest.TestCase):
    def setUp(self):
        self.mask = torch.zeros(1, 1, 9, 9)
        self.mask[..., 3:6, 3:6] = 1.0

    def assert_partition(self, regions):
        total = sum(regions.values())
        self.assertTrue(torch.all(torch.isfinite(total)))
        self.assertLess(float((total - 1.0).abs().max()), 1e-6)
        for value in regions.values():
            self.assertGreaterEqual(float(value.min()), -1e-6)
            self.assertLessEqual(float(value.max()), 1.0 + 1e-6)

    def test_pool_zero(self):
        cfg = PoolMorphologyConfig(pool_grid_scale=1, inner_pool_steps=0, outer_pool_steps=0)
        regions = build_pool_regions(self.mask, cfg)
        self.assertTrue(torch.equal(regions["mask_core"], self.mask))
        self.assertEqual(float(regions["inner_ring"].sum()), 0.0)
        self.assertEqual(float(regions["outer_ring"].sum()), 0.0)
        self.assert_partition(regions)

    def test_pool_one_exp11_outer_parity(self):
        cfg = PoolMorphologyConfig(pool_grid_scale=1, inner_pool_steps=0, outer_pool_steps=1)
        regions = build_pool_regions(self.mask, cfg)
        expected_outer = dilate_steps(self.mask, 1) - self.mask
        self.assertEqual(float((regions["outer_ring"] - expected_outer).abs().max()), 0.0)
        self.assert_partition(regions)
        weights = build_region_weight_map(self.mask, cfg, inner_weight=0.0, outer_weight=0.75)
        expected = self.mask + 0.75 * expected_outer + 0.05 * regions["far_outside"]
        self.assertLess(float((weights - expected).abs().max()), 1e-6)

    def test_inner_one_parity(self):
        cfg = PoolMorphologyConfig(pool_grid_scale=1, inner_pool_steps=1, outer_pool_steps=0)
        regions = build_pool_regions(self.mask, cfg)
        expected_core = erode_steps(self.mask, 1)
        expected_inner = self.mask - expected_core
        self.assertEqual(float((regions["mask_core"] - expected_core).abs().max()), 0.0)
        self.assertEqual(float((regions["inner_ring"] - expected_inner).abs().max()), 0.0)
        self.assert_partition(regions)

    def test_both_one_parity(self):
        cfg = PoolMorphologyConfig(pool_grid_scale=1, inner_pool_steps=1, outer_pool_steps=1)
        regions = build_pool_regions(self.mask, cfg)
        occupied = regions["mask_core"] + regions["inner_ring"] + regions["outer_ring"]
        expected = dilate_steps(self.mask, 1)
        self.assertEqual(float((occupied - expected).abs().max()), 0.0)
        self.assert_partition(regions)

    def test_asymmetric_inner_outer(self):
        cfg = PoolMorphologyConfig(pool_grid_scale=1, inner_pool_steps=2, outer_pool_steps=8)
        regions = build_pool_regions(self.mask, cfg)
        self.assert_partition(regions)
        self.assertTrue(torch.all(torch.isfinite(sum(regions.values()))))

    def test_pool_grid_scale_partition(self):
        for scale in (1, 2, 4):
            cfg = PoolMorphologyConfig(pool_grid_scale=scale, inner_pool_steps=1, outer_pool_steps=2)
            regions = build_pool_regions(self.mask, cfg)
            self.assertEqual(regions["mask_core"].shape, self.mask.shape)
            self.assert_partition(regions)

    def test_empty_region_and_small_mask(self):
        small = torch.zeros(1, 1, 5, 5)
        small[..., 2, 2] = 1.0
        regions = build_pool_regions(small, PoolMorphologyConfig(inner_pool_steps=3, outer_pool_steps=1))
        self.assert_partition(regions)
        self.assertEqual(float(regions["mask_core"].sum()), 0.0)

    def test_region_balanced(self):
        regions = build_pool_regions(self.mask, PoolMorphologyConfig(inner_pool_steps=1, outer_pool_steps=1))
        error = torch.ones_like(self.mask)
        lw = legacy_weighted_mean(error, regions)
        rb = region_balanced_mean(error, regions)
        self.assertTrue(torch.allclose(lw, torch.ones_like(lw)))
        self.assertTrue(torch.allclose(rb, torch.ones_like(rb)))


if __name__ == "__main__":
    unittest.main()

