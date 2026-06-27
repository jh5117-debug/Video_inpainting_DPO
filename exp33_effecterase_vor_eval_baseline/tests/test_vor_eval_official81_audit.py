import unittest

from exp33_effecterase_vor_eval_baseline.scripts.audit_vor_eval_official81 import (
    build_triplets,
    role_from_member,
    strip_vor_eval_prefix,
)


class TestVOREvalOfficial81Audit(unittest.TestCase):
    def test_role_from_member(self):
        self.assertEqual(role_from_member("VOR-Eval/FG_BG/REAL_ENV900_00001_001_03.mp4"), "condition")
        self.assertEqual(role_from_member("VOR-Eval/BG/REAL_ENV900_00001_001_03.mp4"), "winner")
        self.assertEqual(role_from_member("VOR-Eval/MASK/REAL_ENV900_00001_001_03.mp4"), "mask")

    def test_strip_vor_eval_prefix(self):
        self.assertEqual(
            strip_vor_eval_prefix("VOR-Eval/FG_BG/REAL_ENV900_00001_001_03.mp4"),
            "FG_BG/REAL_ENV900_00001_001_03.mp4",
        )
        self.assertEqual(strip_vor_eval_prefix("FG_BG/x.mp4"), "FG_BG/x.mp4")

    def test_build_triplets_requires_all_three_roles(self):
        rows = [
            {
                "type": "file",
                "member_path": "VOR-Eval/BG/REAL_ENV900_00001_001_03.mp4",
            },
            {
                "type": "file",
                "member_path": "VOR-Eval/FG_BG/REAL_ENV900_00001_001_03.mp4",
            },
            {
                "type": "file",
                "member_path": "VOR-Eval/MASK/REAL_ENV900_00001_001_03.mp4",
            },
            {
                "type": "file",
                "member_path": "VOR-Eval/BG/REAL_ENV900_00002_002_05.mp4",
            },
        ]
        triplets = build_triplets(rows)
        self.assertEqual(len(triplets), 1)
        self.assertEqual(triplets[0]["sample_id"], "REAL_ENV900_00001_001_03")
        self.assertEqual(triplets[0]["condition_member_path"], "FG_BG/REAL_ENV900_00001_001_03.mp4")
        self.assertEqual(triplets[0]["winner_member_path"], "BG/REAL_ENV900_00001_001_03.mp4")
        self.assertEqual(triplets[0]["mask_member_path"], "MASK/REAL_ENV900_00001_001_03.mp4")


if __name__ == "__main__":
    unittest.main()
