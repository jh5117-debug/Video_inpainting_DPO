import unittest
from pathlib import Path

from exp33_effecterase_vor_eval_baseline.scripts.audit_vor_eval_official81 import (
    build_triplets,
    role_from_member,
    strip_vor_eval_prefix,
)
from exp33_effecterase_vor_eval_baseline.scripts.materialize_vor_eval_official81_inputs import validate_exp33_row
from exp33_effecterase_vor_eval_baseline.scripts.run_effecterase_vor_eval_official81 import (
    validate_exp33_row as validate_effecterase_run_row,
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

    def test_materializer_requires_held_out_vor_eval_role(self):
        self.assertEqual(
            validate_exp33_row(
                {
                    "vor_eval": True,
                    "eligible_for_training": False,
                    "source_role": "held_out_vor_eval_baseline",
                }
            ),
            [],
        )
        errors = validate_exp33_row(
            {
                "vor_eval": False,
                "eligible_for_training": True,
                "source_role": "diagnostic_only",
            }
        )
        self.assertIn("row_not_marked_vor_eval", errors)
        self.assertIn("training_eligible_row_not_allowed", errors)
        self.assertIn("source_role_not_held_out_vor_eval_baseline", errors)

    def test_effecterase_runner_requires_held_out_baseline_only(self):
        output_root = "/tmp/exp33/output_root"
        self.assertEqual(
            validate_effecterase_run_row(
                {
                    "vor_eval": True,
                    "eligible_for_training": False,
                    "source_role": "held_out_vor_eval_baseline",
                    "scientific_role": "held_out_baseline_only_not_training",
                    "raw_output_primary": True,
                    "output_path": f"{output_root}/outputs/sample/raw_output.mp4",
                },
                Path(output_root),
            ),
            [],
        )
        errors = validate_effecterase_run_row(
            {
                "vor_eval": False,
                "eligible_for_training": True,
                "source_role": "train_candidate",
                "scientific_role": "adapter_training",
                "raw_output_primary": False,
                "output_path": "/tmp/other_root/raw_output.mp4",
            },
            Path(output_root),
        )
        self.assertIn("row_not_marked_vor_eval", errors)
        self.assertIn("training_eligible_row_not_allowed", errors)
        self.assertIn("source_role_not_held_out_vor_eval_baseline", errors)
        self.assertIn("scientific_role_not_baseline_only", errors)
        self.assertIn("raw_output_not_primary", errors)
        self.assertIn("output_path_outside_output_root", errors)


if __name__ == "__main__":
    unittest.main()
