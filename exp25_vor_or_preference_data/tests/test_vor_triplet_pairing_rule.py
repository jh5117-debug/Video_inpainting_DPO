import unittest

from exp25_vor_or_preference_data.scripts.build_vor_triplet_index import role_from_path, scene_group_from_id


class TestVORTripletPairingRule(unittest.TestCase):
    def test_roles_from_real_paths(self):
        self.assertEqual(role_from_path("VOR-Train/FG_BG/REAL_ENV900_00001_001_03.mp4"), "FG_BG")
        self.assertEqual(role_from_path("VOR-Train/BG/REAL_ENV900_00001_001_03.mp4"), "BG")
        self.assertEqual(role_from_path("MASK/REAL_ENV900_00001_001_03.mp4"), "MASK")

    def test_scene_group_real_and_blender(self):
        self.assertEqual(scene_group_from_id("REAL_ENV900_00001_001_03"), "REAL_ENV900_00001")
        self.assertEqual(scene_group_from_id("BLENDER_BEACH038_06047"), "BLENDER_BEACH038")


if __name__ == "__main__":
    unittest.main()
