import unittest

from exp25_vor_or_preference_data.code.dataset.vor_or_manifest_schema import VorORPreferenceRow


class TestORConditionComesFromVObj(unittest.TestCase):
    def test_valid_condition_role(self):
        row = VorORPreferenceRow(
            sample_id="s1",
            split="train",
            task="object_removal",
            condition_video_path="/data/V_obj/s1",
            winner_video_path="/data/V_bg/s1",
            loser_video_path="/data/loser/s1",
            mask_path="/data/mask/s1",
            generator_source="DiffuEraser",
        )
        self.assertNotIn("condition_must_come_from_v_obj", row.validate())

    def test_reject_winner_derived_condition_role(self):
        row = VorORPreferenceRow(
            sample_id="s1",
            split="train",
            task="object_removal",
            condition_video_path="/data/V_bg/s1",
            winner_video_path="/data/V_bg/s1",
            loser_video_path="/data/loser/s1",
            mask_path="/data/mask/s1",
            condition_source_role="V_bg",
            generator_source="DiffuEraser",
        )
        errors = row.validate()
        self.assertIn("condition_must_come_from_v_obj", errors)
        self.assertIn("condition_path_equals_winner_path_gt_leakage_risk", errors)


if __name__ == "__main__":
    unittest.main()
