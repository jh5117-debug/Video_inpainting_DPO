import unittest

import torch
from torch import nn

from exp36_minimax_objective_rescue.code.lora_scope import LoRALinear


class LoraForwardUsageTests(unittest.TestCase):
    def test_lora_residual_changes_forward_when_weight_nonzero(self):
        torch.manual_seed(7)
        base = nn.Linear(4, 3, bias=False)
        wrapped = LoRALinear(base, rank=2, alpha=4)
        x = torch.randn(5, 4)
        y0 = wrapped(x).detach().clone()
        with torch.no_grad():
            wrapped.lora_b.weight.fill_(0.1)
        y1 = wrapped(x).detach()
        self.assertGreater(float((y1 - y0).abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()

