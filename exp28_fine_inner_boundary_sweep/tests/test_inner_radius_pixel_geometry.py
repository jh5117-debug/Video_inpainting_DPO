import unittest

import torch

from exp28_fine_inner_boundary_sweep.code.inner_boundary_geometry import erode_image_mask


class InnerRadiusPixelGeometryTest(unittest.TestCase):
    def test_radius_two_uses_image_space_pixel_erosion(self):
        image_hole = torch.zeros(1, 1, 1, 12, 12)
        image_hole[..., 3:9, 3:9] = 1.0
        eroded = erode_image_mask(image_hole, radius_px=2)
        expected = torch.zeros_like(image_hole)
        expected[..., 5:7, 5:7] = 1.0

        self.assertTrue(torch.equal(eroded, expected))
        self.assertEqual(float((image_hole - eroded).clamp(0.0, 1.0).sum()), 32.0)


if __name__ == "__main__":
    unittest.main()
