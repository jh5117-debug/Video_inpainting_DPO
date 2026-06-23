import unittest

from exp25_vor_or_preference_data.code.dataset.vor_or_manifest_schema import VorORPreferenceRow


class TestORNoHardComp(unittest.TestCase):
    def test_reject_hard_comp_and_comp_mode(self):
        row = VorORPreferenceRow(
            sample_id="s1",
            split="train",
            task="object_removal",
            condition_video_path="/data/V_obj/s1",
            winner_video_path="/data/V_bg/s1",
            loser_video_path="/data/loser/s1",
            mask_path="/data/mask/s1",
            hard_comp=True,
            comp_mode="hard",
            generator_source="DiffuEraser",
        )
        errors = row.validate()
        self.assertIn("hard_comp_must_be_false", errors)
        self.assertIn("comp_mode_must_be_none", errors)


if __name__ == "__main__":
    unittest.main()
