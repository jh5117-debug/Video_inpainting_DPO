import unittest

from pathlib import Path

from exp26_videopainter_dpo_v2.code.train_videopainter_dpo_adapter import normalize_frame_count, resolve_vp2_frame_count


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

    def test_formal_49_rejects_16_input_frames(self):
        files = [Path(f"{i:05d}.png") for i in range(16)]
        with self.assertRaises(ValueError):
            normalize_frame_count(files, 49)

    def test_plumbing_13_selects_exact_13_from_16(self):
        files = [Path(f"{i:05d}.png") for i in range(16)]
        selected = normalize_frame_count(files, 13, plumbing_only_13f=True)
        self.assertEqual(len(selected), 13)


if __name__ == "__main__":
    unittest.main()
