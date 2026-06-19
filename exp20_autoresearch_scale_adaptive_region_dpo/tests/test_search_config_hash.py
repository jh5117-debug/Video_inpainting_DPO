import unittest

from exp20_autoresearch_scale_adaptive_region_dpo.code.search_controller import TrialConfig


class SearchConfigHashTests(unittest.TestCase):
    def test_non_result_fields_do_not_affect_hash(self):
        base = TrialConfig(
            trial_id="trial_a",
            parent_id="root",
            radius_mode="fixed_image_px",
            radius_value=8,
            boundary_contribution=0.75,
            aggregation="legacy_global_weighted_mean",
            description="first description",
            checkpoint_path="/tmp/a",
            log_path="/tmp/a.log",
            gpu_ids="0,1",
        )
        same_result = TrialConfig(
            trial_id="trial_b",
            parent_id="different_parent",
            radius_mode="fixed_image_px",
            radius_value=8,
            boundary_contribution=0.75,
            aggregation="legacy_global_weighted_mean",
            description="different description",
            checkpoint_path="/tmp/b",
            log_path="/tmp/b.log",
            gpu_ids="2,3",
        )
        self.assertEqual(base.config_hash(), same_result.config_hash())

    def test_result_fields_do_affect_hash(self):
        base = TrialConfig(
            trial_id="trial_a",
            parent_id="root",
            radius_mode="fixed_image_px",
            radius_value=8,
            boundary_contribution=0.75,
            aggregation="legacy_global_weighted_mean",
        )
        changed = TrialConfig(
            trial_id="trial_b",
            parent_id="root",
            radius_mode="fixed_image_px",
            radius_value=12,
            boundary_contribution=0.75,
            aggregation="legacy_global_weighted_mean",
        )
        self.assertNotEqual(base.config_hash(), changed.config_hash())


if __name__ == "__main__":
    unittest.main()
