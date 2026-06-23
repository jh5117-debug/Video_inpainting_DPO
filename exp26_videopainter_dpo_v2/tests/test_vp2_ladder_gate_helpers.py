import unittest

import torch

from exp26_videopainter_dpo_v2.code.run_vp2_l0_l4_gates import state_max_abs_diff


class TestVP2LadderGateHelpers(unittest.TestCase):
    def test_state_max_abs_diff_accepts_identical_state(self):
        module = torch.nn.Linear(2, 2)
        state = module.state_dict()
        self.assertEqual(state_max_abs_diff(state, state), 0.0)

    def test_state_max_abs_diff_detects_changed_weight(self):
        module = torch.nn.Linear(2, 2)
        a = {k: v.clone() for k, v in module.state_dict().items()}
        b = {k: v.clone() for k, v in module.state_dict().items()}
        b["weight"][0, 0] += 0.25
        self.assertAlmostEqual(state_max_abs_diff(a, b), 0.25)


if __name__ == "__main__":
    unittest.main()
