import unittest

import torch

from exp26_videopainter_dpo_v2.code.train_videopainter_dpo_adapter import strict_load_state_dict


class TestVP2CheckpointStrictReload(unittest.TestCase):
    def test_strict_reload_rejects_missing_key(self):
        module = torch.nn.Linear(2, 2)
        state = module.state_dict()
        state.pop("bias")
        with self.assertRaises(RuntimeError):
            strict_load_state_dict(module, state)

    def test_strict_reload_accepts_exact_state(self):
        module = torch.nn.Linear(2, 2)
        strict_load_state_dict(module, module.state_dict())


if __name__ == "__main__":
    unittest.main()
