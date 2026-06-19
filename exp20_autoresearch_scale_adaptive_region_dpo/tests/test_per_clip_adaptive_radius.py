import unittest

import torch

from exp20_autoresearch_scale_adaptive_region_dpo.code.boundary_maps import adaptive_radius_per_clip


class PerClipAdaptiveRadiusTests(unittest.TestCase):
    def test_batch_clips_get_independent_radii(self):
        masks = torch.zeros(2, 3, 48, 48)
        masks[0, :, 20:28, 20:28] = 1
        masks[1, :, 12:36, 12:36] = 1
        stats = adaptive_radius_per_clip(
            masks,
            "adaptive_sqrt_area",
            k=1.0,
            r_min=1.0,
            r_max=64.0,
        )
        self.assertEqual(tuple(stats.radius_px.shape), (2,))
        self.assertGreater(float(stats.radius_px[1]), float(stats.radius_px[0]))
        self.assertFalse(torch.isnan(stats.radius_px).any())

    def test_empty_clip_uses_fallback_without_nan(self):
        masks = torch.zeros(2, 3, 32, 32)
        masks[1, :, 8:24, 8:24] = 1
        stats = adaptive_radius_per_clip(
            masks,
            "adaptive_area_perimeter",
            k=1.0,
            r_min=2.0,
            r_max=48.0,
        )
        self.assertEqual(float(stats.empty_clip[0]), 1.0)
        self.assertEqual(float(stats.radius_px[0]), 2.0)
        self.assertFalse(torch.isnan(stats.radius_px).any())


if __name__ == "__main__":
    unittest.main()
