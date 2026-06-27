import unittest

from exp35_minimax_flow_dpo_rescue.code.scope_audit import TensorScope, module_family, module_group, summarize_tensors


class TrainableScopeNameTests(unittest.TestCase):
    def test_module_group_prefers_transformer_block_prefix(self):
        self.assertEqual(module_group("blocks.3.attn.to_q.weight"), "blocks.3")

    def test_module_family_detects_attention_and_lora(self):
        self.assertEqual(module_family("blocks.0.attn.to_q.weight"), "attention_q")
        self.assertEqual(module_family("blocks.0.attn.to_out.0.lora_A.weight"), "lora_or_adapter")

    def test_summary_counts_lora_markers(self):
        summary = summarize_tensors([
            TensorScope("blocks.0.attn.to_q.weight", (2, 3)),
            TensorScope("blocks.0.attn.to_q.lora_A.weight", (1, 2)),
        ])
        self.assertEqual(summary["tensor_count"], 2)
        self.assertEqual(summary["lora_or_adapter_tensor_count"], 1)


if __name__ == "__main__":
    unittest.main()
