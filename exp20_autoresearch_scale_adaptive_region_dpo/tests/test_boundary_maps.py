import unittest

import torch

from exp20_autoresearch_scale_adaptive_region_dpo.code.boundary_maps import (
    adaptive_radius,
    build_region_maps,
    image_space_outer_ring,
    legacy_latent_outer_ring,
)


class BoundaryMapTests(unittest.TestCase):
    def test_legacy_outer_ring_one_cell(self):
        mask = torch.zeros(1, 1, 5, 5)
        mask[:, :, 2, 2] = 1
        ring = legacy_latent_outer_ring(mask)
        self.assertEqual(float(ring.sum()), 8.0)
        self.assertEqual(float(ring[0, 0, 2, 2]), 0.0)

    def test_image_space_outer_ring_radius(self):
        mask = torch.zeros(1, 1, 7, 7)
        mask[:, :, 3, 3] = 1
        ring = image_space_outer_ring(mask, 1.01)
        self.assertEqual(float(ring.sum()), 4.0)
        self.assertEqual(float(ring[0, 0, 3, 2]), 1.0)
        self.assertEqual(float(ring[0, 0, 2, 2]), 0.0)

    def test_empty_mask_no_nan(self):
        mask = torch.zeros(1, 2, 16, 16)
        maps = build_region_maps(mask, (4, 4), radius_mode="fixed_image_px", radius_value=8)
        self.assertFalse(torch.isnan(maps["boundary"]).any())
        self.assertEqual(float(maps["mask"].sum()), 0.0)

    def test_adaptive_radius_stable_clip_stat(self):
        mask = torch.zeros(1, 3, 20, 20)
        mask[:, :, 5:15, 5:15] = 1
        stats = adaptive_radius(mask, "adaptive_area_perimeter", k=1.0, r_min=2, r_max=48)
        self.assertGreaterEqual(stats.radius_px, 2.0)
        self.assertLessEqual(stats.radius_px, 48.0)
        self.assertGreater(stats.area_median, 0.0)


if __name__ == "__main__":
    unittest.main()
