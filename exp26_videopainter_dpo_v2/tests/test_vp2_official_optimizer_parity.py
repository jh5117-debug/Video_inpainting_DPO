import argparse
from pathlib import Path
import tempfile
import unittest

import torch

from exp26_videopainter_dpo_v2.code.train_videopainter_dpo_adapter import make_vp2_optimizer
from exp26_videopainter_dpo_v2.code.vp2_official_config import parse_official_optimizer_scheduler_config


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

    def test_parse_official_argparse_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "official_train.py"
            path.write_text(
                "import argparse\n"
                "p=argparse.ArgumentParser()\n"
                "p.add_argument('--learning_rate', default=2e-5)\n"
                "p.add_argument('--adam_beta1', default=0.9)\n"
                "p.add_argument('--adam_beta2', default=0.95)\n"
                "p.add_argument('--adam_epsilon', default=1e-8)\n"
                "p.add_argument('--weight_decay', default=0.01)\n"
                "p.add_argument('--lr_scheduler', default='cosine')\n"
                "p.add_argument('--lr_warmup_steps', default=100)\n"
                "p.add_argument('--max_grad_norm', default=1.0)\n"
            )
            cfg = parse_official_optimizer_scheduler_config(path)
            self.assertEqual(cfg.learning_rate, 2e-5)
            self.assertEqual(cfg.adam_beta2, 0.95)
            self.assertEqual(cfg.lr_scheduler, "cosine")
            self.assertEqual(cfg.lr_warmup_steps, 100)


if __name__ == "__main__":
    unittest.main()
