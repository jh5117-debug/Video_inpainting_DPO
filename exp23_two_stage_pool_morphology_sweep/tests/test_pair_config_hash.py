import unittest

from exp23_two_stage_pool_morphology_sweep.code.exp23_trial_runner import (
    ModelPlan,
    RegionConfig,
    RunConfig,
    pair_config_hash,
)


class PairConfigHashTest(unittest.TestCase):
    def test_pair_id_does_not_change_scientific_config_hash(self):
        plans = [
            ModelPlan("fresh", RegionConfig(True, "outer", 1, 0, 1, 0.0, 0.75, 0.75), RegionConfig(True, "outer", 1, 0, 1, 0.0, 0.75, 0.75)),
            ModelPlan("candidate", RegionConfig(False, "outer", 1, 0, 2, 0.0, 0.75, 0.75), RegionConfig(False, "outer", 1, 0, 2, 0.0, 0.75, 0.75)),
        ]
        self.assertEqual(pair_config_hash(RunConfig(pair_id="a"), plans), pair_config_hash(RunConfig(pair_id="b"), plans))

    def test_region_change_changes_hash(self):
        run = RunConfig(pair_id="unit")
        plans_a = [ModelPlan("candidate", RegionConfig(False, "outer", 1, 0, 2, 0.0, 0.75, 0.75), RegionConfig(False, "outer", 1, 0, 2, 0.0, 0.75, 0.75))]
        plans_b = [ModelPlan("candidate", RegionConfig(False, "outer", 1, 0, 3, 0.0, 0.75, 0.75), RegionConfig(False, "outer", 1, 0, 3, 0.0, 0.75, 0.75))]
        self.assertNotEqual(pair_config_hash(run, plans_a), pair_config_hash(run, plans_b))


if __name__ == "__main__":
    unittest.main()
