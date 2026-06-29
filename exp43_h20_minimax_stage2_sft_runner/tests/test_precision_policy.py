import unittest

from exp43_h20_minimax_stage2_sft_runner.precision_policy import (
    PrecisionPolicy,
    clamp_timestep,
    classify_runtime_failure,
)


class PrecisionPolicyTest(unittest.TestCase):
    def test_policy_disallows_silent_fallback(self):
        policy = PrecisionPolicy()
        self.assertFalse(policy.silent_fallback_allowed)
        self.assertEqual(policy.vae_dtype, "fp32")
        self.assertEqual(policy.transformer_dtype, "bf16")

    def test_timestep_clamp_avoids_edges(self):
        policy = PrecisionPolicy(timestep_min=0.05, timestep_max=0.95)
        self.assertEqual(clamp_timestep(0.0, policy), 0.05)
        self.assertEqual(clamp_timestep(1.0, policy), 0.95)
        self.assertEqual(clamp_timestep(0.37, policy), 0.37)

    def test_failure_classification(self):
        self.assertEqual(classify_runtime_failure("CUDA out of memory"), "OOM")
        self.assertEqual(classify_runtime_failure("xFormers kernel failed"), "XFORMERS")
        self.assertEqual(classify_runtime_failure("checkpoint reload nonfinite"), "LOSS_REDUCTION")


if __name__ == "__main__":
    unittest.main()
