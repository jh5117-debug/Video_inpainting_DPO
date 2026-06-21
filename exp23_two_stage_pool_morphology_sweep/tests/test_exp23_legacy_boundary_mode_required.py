import unittest
from pathlib import Path


class Exp23BoundaryModeRequiredTest(unittest.TestCase):
    def test_missing_boundary_mode_is_explicit_error(self):
        stage1 = Path("exp23_two_stage_pool_morphology_sweep/code/train_stage1.py").read_text(encoding="utf-8")
        stage2 = Path("exp23_two_stage_pool_morphology_sweep/code/train_exp23_stage2.py").read_text(encoding="utf-8")
        expected = "Exp23 requires explicit --boundary_mode"
        self.assertIn(expected, stage1)
        self.assertIn(expected, stage2)

    def test_region_diagnostics_are_written(self):
        stage1 = Path("exp23_two_stage_pool_morphology_sweep/code/train_stage1.py").read_text(encoding="utf-8")
        stage2 = Path("exp23_two_stage_pool_morphology_sweep/code/train_exp23_stage2.py").read_text(encoding="utf-8")
        self.assertIn("append_region_diagnostics_csv", stage1)
        self.assertIn("append_region_diagnostics_csv", stage2)
        self.assertIn("resolved_region_config.json", stage1)


if __name__ == "__main__":
    unittest.main()
