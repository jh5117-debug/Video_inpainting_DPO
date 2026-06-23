import argparse
import unittest

import torch

from exp26_videopainter_dpo_v2.code.train_videopainter_dpo_adapter import make_vp2_optimizer


class TestVP2OptimizerConfig(unittest.TestCase):
    def test_optimizer_uses_explicit_official_fields(self):
        param = torch.nn.Parameter(torch.ones(()))
        args = argparse.Namespace(
            learning_rate=3e-5,
            adam_beta1=0.8,
            adam_beta2=0.88,
            adam_epsilon=1e-6,
            weight_decay=0.123,
        )
        opt = make_vp2_optimizer([param], args)
        group = opt.param_groups[0]
        self.assertEqual(group["lr"], 3e-5)
        self.assertEqual(group["betas"], (0.8, 0.88))
        self.assertEqual(group["eps"], 1e-6)
        self.assertEqual(group["weight_decay"], 0.123)


if __name__ == "__main__":
    unittest.main()
