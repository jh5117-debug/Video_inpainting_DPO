import unittest

import torch

from exp20_autoresearch_scale_adaptive_region_dpo.code.boundary_maps import build_region_maps


class RegionPartitionOfUnityTests(unittest.TestCase):
    def test_image_space_maps_partition_each_loss_cell(self):
        mask = torch.zeros(1, 2, 37, 41)
        mask[:, :, 9:24, 11:29] = 1
        maps = build_region_maps(mask, (7, 9), radius_mode="fixed_image_px", radius_value=5)
        total = maps["mask"] + maps["boundary"] + maps["outside"]
        self.assertGreater(float(maps["boundary"].sum()), 0.0)
        self.assertTrue(torch.all(maps["mask"] >= 0))
        self.assertTrue(torch.all(maps["boundary"] >= 0))
        self.assertTrue(torch.all(maps["outside"] >= 0))
        self.assertLess(float((total - 1.0).abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
