import unittest

from exp26_videopainter_dpo_v2.code.train_videopainter_dpo_adapter import resolve_vp2_frame_count


class TestVP2FrameCount(unittest.TestCase):
    def test_native_49_allowed(self):
        self.assertEqual(resolve_vp2_frame_count(49), 49)

    def test_13_requires_plumbing_flag(self):
        with self.assertRaises(ValueError):
            resolve_vp2_frame_count(13)
        self.assertEqual(resolve_vp2_frame_count(13, plumbing_only_13f=True), 13)

    def test_non_4k_plus_1_rejected(self):
        with self.assertRaises(ValueError):
            resolve_vp2_frame_count(16)


if __name__ == "__main__":
    unittest.main()
