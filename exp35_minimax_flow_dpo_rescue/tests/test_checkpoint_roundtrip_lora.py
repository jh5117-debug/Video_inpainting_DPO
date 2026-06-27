import unittest

from exp35_minimax_flow_dpo_rescue.code.scope_audit import TensorScope, summarize_tensors


class CheckpointRoundtripScopeTests(unittest.TestCase):
    def test_no_lora_scope_is_explicit(self):
        summary = summarize_tensors([TensorScope("blocks.0.ffn.weight", (4, 4))])
        self.assertEqual(summary["lora_or_adapter_tensor_count"], 0)


if __name__ == "__main__":
    unittest.main()
