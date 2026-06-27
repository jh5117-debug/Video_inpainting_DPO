import unittest

import torch
from torch import nn

from exp36_minimax_objective_rescue.code.lora_scope import LoRALinear


class CheckpointRoundtripTests(unittest.TestCase):
    def test_lora_state_dict_roundtrip_is_exact(self):
        torch.manual_seed(11)
        original = LoRALinear(nn.Linear(4, 3, bias=False), rank=2, alpha=4)
        with torch.no_grad():
            original.lora_b.weight.fill_(0.25)
        clone = LoRALinear(nn.Linear(4, 3, bias=False), rank=2, alpha=4)
        missing, unexpected = clone.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, clone.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()

