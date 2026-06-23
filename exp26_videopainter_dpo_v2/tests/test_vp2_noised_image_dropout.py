import unittest

import torch

from exp26_videopainter_dpo_v2.code.train_videopainter_dpo_adapter import apply_noised_image_dropout


class TestVP2NoisedImageDropout(unittest.TestCase):
    def test_dropout_one_zeroes_images(self):
        x = torch.ones(2, 3, 1, 4, 4)
        y = apply_noised_image_dropout(x, 1.0, training=True)
        self.assertTrue(torch.equal(y, torch.zeros_like(x)))

    def test_dropout_disabled_in_eval(self):
        x = torch.ones(2, 3, 1, 4, 4)
        y = apply_noised_image_dropout(x, 1.0, training=False)
        self.assertTrue(torch.equal(y, x))


if __name__ == "__main__":
    unittest.main()
