import unittest

from exp36_minimax_objective_rescue.code.scope_audit import (
    TensorScope,
    exp36_scope_plans,
    module_family,
    module_group,
    scope_selects_tensor,
    summarize_tensors,
)


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

    def test_scope_plans_select_expected_families(self):
        plans = exp36_scope_plans()
        self.assertTrue(scope_selects_tensor(plans["S1"], "blocks.0.attn.to_q.weight"))
        self.assertFalse(scope_selects_tensor(plans["S1"], "blocks.0.ffn.net.0.proj.weight"))
        self.assertTrue(scope_selects_tensor(plans["S2"], "blocks.28.ffn.net.0.proj.weight", block_index=28))
        self.assertFalse(scope_selects_tensor(plans["S2"], "blocks.2.ffn.net.0.proj.weight", block_index=2))


if __name__ == "__main__":
    unittest.main()

