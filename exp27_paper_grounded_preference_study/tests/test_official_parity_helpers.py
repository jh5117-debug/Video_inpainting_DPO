import unittest

import torch

from exp27_paper_grounded_preference_study.code.official_parity import (
    ema_update_tensor,
    exp27_sdpo_safe_lambda,
    linear_dpo_clip_ratio,
    localdpo_mask_digest,
)


class TestOfficialParityHelpers(unittest.TestCase):
    def test_sdpo_lambda_shape_and_range(self):
        pred = torch.randn(4, 2, 3, 3)
        target = torch.randn_like(pred)
        lam = exp27_sdpo_safe_lambda(pred, target, mu=0.2)
        self.assertEqual(tuple(lam.shape), ())
        self.assertGreaterEqual(float(lam.item()), 0.0)
        self.assertLessEqual(float(lam.item()), 1.0)

    def test_linear_dpo_ratio_formula(self):
        ratio = linear_dpo_clip_ratio(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]), beta_dpo=5000, eta_dpo=0.01)
        self.assertTrue(torch.allclose(ratio, torch.tensor([0.5, 0.5])))
        clipped = linear_dpo_clip_ratio(torch.tensor([1.0]), torch.tensor([0.0]), beta_dpo=5000, eta_dpo=0.01)
        self.assertAlmostEqual(float(clipped.item()), 0.99, places=6)

    def test_ema_update_tensor(self):
        ema = torch.tensor([1.0])
        model = torch.tensor([3.0])
        self.assertTrue(torch.allclose(ema_update_tensor(ema, model, 0.9), torch.tensor([1.2])))

    def test_localdpo_mask_deterministic(self):
        a = localdpo_mask_digest(seed=123, video_length=4, image_height=64, image_width=96)
        b = localdpo_mask_digest(seed=123, video_length=4, image_height=64, image_width=96)
        self.assertEqual(a, b)
        if a.get("status") == "blocked_official_code_runtime_error":
            self.assertIn("random_mask_gen.py", a["source"])
            self.assertIn("reshape", a["error"])
            return
        self.assertEqual(a["shape"], [4, 64, 96])


if __name__ == "__main__":
    unittest.main()
