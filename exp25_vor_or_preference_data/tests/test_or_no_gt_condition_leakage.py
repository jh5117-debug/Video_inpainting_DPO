import unittest

from exp25_vor_or_preference_data.code.dataset.vor_or_manifest_schema import VorORPreferenceRow


class TestORNoGTConditionLeakage(unittest.TestCase):
    def test_condition_path_must_not_equal_winner_path(self):
        row = VorORPreferenceRow(
            sample_id="s1",
            split="train",
            task="object_removal",
            condition_video_path="/data/V_bg/s1",
            winner_video_path="/data/V_bg/s1",
            loser_video_path="/data/loser/s1",
            mask_path="/data/mask/s1",
            generator_source="DiffuEraser",
        )
        self.assertIn("condition_path_equals_winner_path_gt_leakage_risk", row.validate())


if __name__ == "__main__":
    unittest.main()
