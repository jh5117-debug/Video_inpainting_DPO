import unittest

from exp60b_videopainter_vpdata_d3mask.vpdata_subset import classify_path


class VPDataSubsetTests(unittest.TestCase):
    def test_classify_pexels_path(self):
        kind, index, video_id = classify_path("000000004397_2256073.mp4")
        self.assertEqual(kind, "pexels")
        self.assertEqual(index, 4397)
        self.assertEqual(video_id, "2256073")

    def test_classify_videovo_path(self):
        kind, index, video_id = classify_path("000005000000.0.mp4")
        self.assertEqual(kind, "videovo")
        self.assertEqual(index, 5000000)
        self.assertEqual(video_id, "000005000000")

    def test_classify_unknown_path(self):
        kind, index, video_id = classify_path("bad/path.mp4")
        self.assertEqual(kind, "unknown")
        self.assertIsNone(index)
        self.assertEqual(video_id, "bad/path.mp4")


if __name__ == "__main__":
    unittest.main()

