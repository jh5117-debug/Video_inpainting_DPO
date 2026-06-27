import unittest

from torch import nn

from exp36_minimax_objective_rescue.code.lora_scope import LoRALinear, trainable_parameter_names


class ReferenceFrozenContractTests(unittest.TestCase):
    def test_base_reference_weights_are_frozen_in_lora_wrapper(self):
        wrapped = LoRALinear(nn.Linear(4, 3, bias=False), rank=2, alpha=4)
        names = trainable_parameter_names(wrapped)
        self.assertEqual(set(names), {"lora_a.weight", "lora_b.weight"})
        self.assertFalse(wrapped.base.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()

