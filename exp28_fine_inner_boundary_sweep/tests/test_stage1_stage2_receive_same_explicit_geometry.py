import unittest
from pathlib import Path

from exp28_fine_inner_boundary_sweep.code.exp28_trial_runner import (
    REFERENCE_EFFECTIVE_BATCH,
    REFERENCE_PER_DEVICE_BATCH,
    RegionConfig,
    RunConfig,
    adjusted_grad_accum,
    base_args,
)


def _option_value(args, option):
    return args[args.index(option) + 1]


class Stage1Stage2ReceiveSameExplicitGeometryTest(unittest.TestCase):
    def test_stage1_stage2_have_same_explicit_inner_geometry(self):
        region = RegionConfig("inner_boundary_px", 4)
        run = RunConfig(pair_id="unit")
        s1 = base_args(Path("/tmp/exp28-s1"), run, region, stage=1, world_size=2)
        s2 = base_args(
            Path("/tmp/exp28-s2"),
            run,
            region,
            stage=2,
            world_size=2,
            stage1_weights=Path("/tmp/exp28-s1/last_weights"),
        )

        for args in (s1, s2):
            self.assertEqual(_option_value(args, "--exp28_geometry_mode"), "inner_boundary_px")
            self.assertEqual(_option_value(args, "--inner_radius_px"), "4")
            self.assertEqual(_option_value(args, "--legacy_exact"), "false")
            self.assertEqual(_option_value(args, "--outer_pool_steps"), "1")
            self.assertEqual(_option_value(args, "--gradient_accumulation_steps"), "2")

        effective = REFERENCE_PER_DEVICE_BATCH * adjusted_grad_accum(2) * 2
        self.assertEqual(effective, REFERENCE_EFFECTIVE_BATCH)


if __name__ == "__main__":
    unittest.main()
