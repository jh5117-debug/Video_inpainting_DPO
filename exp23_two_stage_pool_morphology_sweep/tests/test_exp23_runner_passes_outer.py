import unittest

from exp23_two_stage_pool_morphology_sweep.code.exp23_trial_runner import RegionConfig, RunConfig, base_args


def _option_value(args, option):
    idx = args.index(option)
    return args[idx + 1]


class Exp23RunnerPassesOuterTest(unittest.TestCase):
    def test_runner_fresh_control_passes_outer_boundary_mode(self):
        region = RegionConfig("fresh_exp11_outer_b075", True, "outer", 1, 0, 1, 0.0, 0.75, 0.75)
        args = base_args("/tmp/fresh", RunConfig(pair_id="unit"), region, stage=1)
        self.assertEqual(_option_value(args, "--boundary_mode"), "outer")
        self.assertEqual(_option_value(args, "--legacy_exact"), "true")
        self.assertEqual(_option_value(args, "--outer_pool_steps"), "1")

    def test_runner_candidate_passes_outer_boundary_mode_and_outer2(self):
        region = RegionConfig("candidate_scale1_outer2_b075", False, "outer", 1, 0, 2, 0.0, 0.75, 0.75)
        args = base_args("/tmp/candidate", RunConfig(pair_id="unit"), region, stage=1)
        self.assertEqual(_option_value(args, "--boundary_mode"), "outer")
        self.assertEqual(_option_value(args, "--legacy_exact"), "false")
        self.assertEqual(_option_value(args, "--outer_pool_steps"), "2")


if __name__ == "__main__":
    unittest.main()
