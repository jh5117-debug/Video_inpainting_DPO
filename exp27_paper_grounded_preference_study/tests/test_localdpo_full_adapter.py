import unittest

import torch

from exp27_paper_grounded_preference_study.code.localdpo_full_adapter import (
    LocalDpoMasks,
    localdpo_latent_fusion,
    progressive_outside_reinjection,
    region_aware_l1,
)


class TestLocalDpoFullAdapter(unittest.TestCase):
    def test_latent_fusion_preserves_outside_original(self):
        denoised = torch.ones(1, 4, 2, 8, 8)
        original = torch.zeros_like(denoised)
        mask = torch.zeros(1, 4, 1, 8, 8)
        mask[:, :, :, 2:6, 3:7] = 1.0
        fused = localdpo_latent_fusion(denoised, original, mask)
        self.assertEqual(float(fused[:, :, :, 0:2, :].sum().item()), 0.0)
        self.assertGreater(float(fused[:, :, :, 2:6, 3:7].sum().item()), 0.0)

    def test_progressive_reinjection_matches_fusion(self):
        current = torch.randn(1, 2, 1, 4, 4)
        step = torch.randn_like(current)
        original = torch.randn_like(current)
        mask = torch.zeros(1, 2, 1, 4, 4)
        mask[..., 1:3, 1:3] = 1.0
        a = progressive_outside_reinjection(current, step, original, mask)
        b = localdpo_latent_fusion(step, original, mask)
        self.assertTrue(torch.allclose(a, b))

    def test_masks_are_distinct_and_validated(self):
        task = torch.zeros(1, 3, 1, 5, 5)
        corruption = torch.zeros_like(task)
        restoration = torch.zeros_like(task)
        task[..., 1:4, 1:4] = 1.0
        corruption[..., 2:5, 2:5] = 1.0
        restoration[..., 1:5, 1:5] = 1.0
        bundle = LocalDpoMasks(task, corruption, restoration)
        bundle.validate()
        self.assertFalse(torch.equal(bundle.task_mask, bundle.corruption_mask))
        self.assertFalse(torch.equal(bundle.restoration_region, bundle.task_mask))

    def test_region_aware_l1_uses_restoration_region(self):
        pred = torch.ones(1, 1, 2, 4, 4)
        target = torch.zeros_like(pred)
        region = torch.zeros(1, 1, 1, 4, 4)
        region[..., :2, :2] = 1.0
        loss = region_aware_l1(pred, target, region)
        self.assertAlmostEqual(float(loss.item()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
