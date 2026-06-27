import argparse
from pathlib import Path
import tempfile
import unittest

import torch

from exp26_videopainter_dpo_v2.code.train_videopainter_dpo_adapter import (
    VideoPainterDPOTrainer,
    parse_checkpoint_steps,
    should_save_checkpoint,
)


class FakeBranch:
    def save_pretrained(self, path, **kwargs):
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "weights.txt").write_text("fake", encoding="utf-8")


class TestVP2ExplicitCheckpoints(unittest.TestCase):
    def test_parse_checkpoint_steps_accepts_zero_and_sorts_by_set(self):
        self.assertEqual(parse_checkpoint_steps("0,1,10,10", 20), {0, 1, 10})

    def test_parse_checkpoint_steps_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            parse_checkpoint_steps("0,21", 20)

    def test_should_save_checkpoint_combines_periodic_and_explicit(self):
        explicit = {1, 10}
        self.assertTrue(should_save_checkpoint(1, 500, explicit))
        self.assertTrue(should_save_checkpoint(500, 500, explicit))
        self.assertFalse(should_save_checkpoint(2, 500, explicit))
        self.assertTrue(should_save_checkpoint(10, 0, explicit))

    def test_retention_preserves_explicit_checkpoint_steps(self):
        with tempfile.TemporaryDirectory() as td:
            trainer = object.__new__(VideoPainterDPOTrainer)
            trainer.args = argparse.Namespace(output_dir=td, checkpoints_total_limit=3)
            trainer.policy_branch = FakeBranch()
            param = torch.nn.Parameter(torch.ones(()))
            optimizer = torch.optim.SGD([param], lr=0.1)

            for step in (0, 1, 2, 3):
                trainer.save_branch_checkpoint(step, optimizer, protected_steps={0, 1})

            names = sorted(p.name for p in Path(td).glob("checkpoint-*"))
            self.assertIn("checkpoint-0", names)
            self.assertIn("checkpoint-1", names)
            self.assertNotIn("checkpoint-2", names)
            self.assertIn("checkpoint-3", names)


if __name__ == "__main__":
    unittest.main()
