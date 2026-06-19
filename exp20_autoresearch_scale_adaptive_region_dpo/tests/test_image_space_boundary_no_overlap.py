import unittest

import torch

from exp20_autoresearch_scale_adaptive_region_dpo.code.boundary_maps import image_space_outer_ring


class ImageSpaceBoundaryNoOverlapTests(unittest.TestCase):
    def test_outer_boundary_excludes_hole_pixels(self):
        mask = torch.zeros(1, 1, 13, 13)
        mask[:, :, 5:8, 5:8] = 1
        boundary = image_space_outer_ring(mask, 2.0)
        overlap = boundary * mask
        self.assertEqual(float(overlap.sum()), 0.0)
        self.assertGreater(float(boundary.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
