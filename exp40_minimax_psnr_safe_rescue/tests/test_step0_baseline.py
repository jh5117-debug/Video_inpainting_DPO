import unittest

from exp40_minimax_psnr_safe_rescue.scripts import run_step0_baseline as baseline


class Step0BaselineTests(unittest.TestCase):
    def test_parse_split_manifest(self):
        parsed = baseline.parse_split_manifest(["train=/tmp/a.jsonl", "shadow=/tmp/b.jsonl"])
        self.assertEqual(parsed[0][0], "train")
        self.assertTrue(str(parsed[1][1]).endswith("/tmp/b.jsonl"))

    def test_aggregate_by_split(self):
        rows = [
            {"split": "train", "full_psnr": 1.0, "mask_psnr": 2.0},
            {"split": "train", "full_psnr": 3.0, "mask_psnr": 4.0},
            {"split": "search", "full_psnr": 5.0, "mask_psnr": 6.0},
        ]
        out = baseline.aggregate_by_split(rows)
        self.assertEqual(out["train"]["rows"], 2)
        self.assertEqual(out["train"]["full_psnr"], 2.0)
        self.assertEqual(out["search"]["mask_psnr"], 6.0)


if __name__ == "__main__":
    unittest.main()
