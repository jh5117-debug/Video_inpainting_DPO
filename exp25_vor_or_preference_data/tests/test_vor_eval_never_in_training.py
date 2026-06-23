import unittest

from exp25_vor_or_preference_data.code.dataset.vor_or_manifest_schema import VorORTripletRow


class TestVOREvalNeverInTraining(unittest.TestCase):
    def test_reject_vor_eval_split_by_default(self):
        row = VorORTripletRow(
            sample_id="eval001",
            split="VOR-Eval",
            task="object_removal",
            condition_video_path="/data/V_obj/eval001",
            winner_video_path="/data/V_bg/eval001",
            mask_path="/data/mask/eval001",
        )
        self.assertIn("vor_eval_must_not_enter_training_or_dev_manifests", row.validate())

    def test_allow_eval_only_when_explicit(self):
        row = VorORTripletRow(
            sample_id="eval001",
            split="VOR-Eval",
            task="object_removal",
            condition_video_path="/data/V_obj/eval001",
            winner_video_path="/data/V_bg/eval001",
            mask_path="/data/mask/eval001",
        )
        self.assertNotIn("vor_eval_must_not_enter_training_or_dev_manifests", row.validate(allow_eval=True))


if __name__ == "__main__":
    unittest.main()
