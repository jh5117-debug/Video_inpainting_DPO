import tempfile
import unittest
from pathlib import Path

from exp20_autoresearch_scale_adaptive_region_dpo.code.search_controller import SearchController, TrialConfig


class SearchControllerTests(unittest.TestCase):
    def test_config_hash_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctl = SearchController(Path(tmp))
            cfg = TrialConfig(
                trial_id="t1",
                parent_id="",
                radius_mode="fixed_image_px",
                radius_value=8,
                boundary_contribution=0.75,
                aggregation="legacy_global_weighted_mean",
            )
            self.assertIsNotNone(ctl.enqueue(cfg))
            cfg2 = TrialConfig(
                trial_id="t2",
                parent_id="",
                radius_mode="fixed_image_px",
                radius_value=8,
                boundary_contribution=0.75,
                aggregation="legacy_global_weighted_mean",
            )
            self.assertIsNone(ctl.enqueue(cfg2))


if __name__ == "__main__":
    unittest.main()
