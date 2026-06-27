import unittest

from exp35_minimax_flow_dpo_rescue.code.scope_audit import has_lora_marker


class LoraModuleMarkerTests(unittest.TestCase):
    def test_marker_detection_is_case_insensitive(self):
        self.assertTrue(has_lora_marker("Transformer.Block.LoRA_A.weight"))
        self.assertTrue(has_lora_marker("adapter.down.weight"))
        self.assertFalse(has_lora_marker("blocks.0.attn.to_q.weight"))


if __name__ == "__main__":
    unittest.main()
