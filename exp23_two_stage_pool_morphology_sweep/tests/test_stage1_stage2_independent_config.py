import unittest
from pathlib import Path

from exp23_two_stage_pool_morphology_sweep.code.exp23_trial_runner import (
    ModelPlan,
    RegionConfig,
    RunConfig,
    base_args,
)


def _option_value(args, option):
    idx = args.index(option)
    return args[idx + 1]


class StageIndependentConfigTest(unittest.TestCase):
    def test_stage1_and_stage2_can_use_different_region_configs(self):
        plan = ModelPlan(
            "candidate",
            stage1_region=RegionConfig(False, "outer", 1, 0, 2, 0.0, 0.75, 0.75),
            stage2_region=RegionConfig(False, "outer", 1, 0, 1, 0.0, 0.50, 0.50),
        )
        run = RunConfig(pair_id="unit")
        s1_args = base_args(Path("/tmp/exp23-stage1"), run, plan.stage1_region, stage=1)
        s2_args = base_args(
            Path("/tmp/exp23-stage2"),
            run,
            plan.stage2_region,
            stage=2,
            stage1_weights=Path("/tmp/exp23-stage1/last_weights"),
        )
        self.assertEqual(_option_value(s1_args, "--outer_pool_steps"), "2")
        self.assertEqual(_option_value(s1_args, "--outer_weight"), "0.75")
        self.assertEqual(_option_value(s2_args, "--outer_pool_steps"), "1")
        self.assertEqual(_option_value(s2_args, "--outer_weight"), "0.5")


if __name__ == "__main__":
    unittest.main()
