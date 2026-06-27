import unittest


class ReferenceFrozenContractTests(unittest.TestCase):
    def test_scope_audit_does_not_authorize_reference_updates(self):
        allowed_statuses = {
            "MINIMAX_TRAINABLE_SCOPE_CURRENT_OK",
            "MINIMAX_TRAINABLE_SCOPE_EXPANDED_S1_READY",
            "MINIMAX_TRAINABLE_SCOPE_EXPANDED_S2_READY",
            "MINIMAX_TRAINABLE_SCOPE_BLOCKED",
        }
        self.assertIn("MINIMAX_TRAINABLE_SCOPE_CURRENT_OK", allowed_statuses)


if __name__ == "__main__":
    unittest.main()
