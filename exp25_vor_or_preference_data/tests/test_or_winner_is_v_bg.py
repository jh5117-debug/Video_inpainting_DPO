import unittest

from exp25_vor_or_preference_data.code.dataset.vor_or_manifest_schema import VorORTripletRow


class TestORWinnerIsVBg(unittest.TestCase):
    def test_reject_non_bg_winner_role(self):
        row = VorORTripletRow(
            sample_id="s1",
            split="search_dev",
            task="object_removal",
            condition_video_path="/data/V_obj/s1",
            winner_video_path="/data/V_obj/s1",
            mask_path="/data/mask/s1",
            winner_source_role="V_obj",
        )
        errors = row.validate()
        self.assertIn("winner_must_be_v_bg", errors)


if __name__ == "__main__":
    unittest.main()
