import unittest
import json
import tempfile
from pathlib import Path

from exp25_vor_or_preference_data.scripts.build_vor_triplet_index import role_from_path, scene_group_from_id
from exp25_vor_or_preference_data.scripts.safe_extract_vor_subset import load_target_member_paths


class TestVORTripletPairingRule(unittest.TestCase):
    def test_roles_from_real_paths(self):
        self.assertEqual(role_from_path("VOR-Train/FG_BG/REAL_ENV900_00001_001_03.mp4"), "FG_BG")
        self.assertEqual(role_from_path("VOR-Train/BG/REAL_ENV900_00001_001_03.mp4"), "BG")
        self.assertEqual(role_from_path("MASK/REAL_ENV900_00001_001_03.mp4"), "MASK")

    def test_scene_group_real_and_blender(self):
        self.assertEqual(scene_group_from_id("REAL_ENV900_00001_001_03"), "REAL_ENV900_00001")
        self.assertEqual(scene_group_from_id("BLENDER_BEACH038_06047"), "BLENDER_BEACH038")

    def test_triplet_jsonl_exact_member_targets(self):
        row = {
            "condition_member_path": "VOR-Train/FG_BG/REAL_ENV900_00001_001_03.mp4",
            "winner_member_path": "VOR-Train/BG/REAL_ENV900_00001_001_03.mp4",
            "mask_member_path": "MASK/REAL_ENV900_00001_001_03.mp4",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "triplets.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            targets = load_target_member_paths(path)
        self.assertEqual(
            targets["VOR-Train"],
            {
                "VOR-Train/FG_BG/REAL_ENV900_00001_001_03.mp4",
                "VOR-Train/BG/REAL_ENV900_00001_001_03.mp4",
            },
        )
        self.assertEqual(targets["VOR-Train-MASK"], {"MASK/REAL_ENV900_00001_001_03.mp4"})


if __name__ == "__main__":
    unittest.main()
