import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_sft_psnr_safe_grid.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_sft_psnr_safe_grid", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SftPsnrSafeGridTest(unittest.TestCase):
    def test_recipe_weights_match_preregistered_values(self):
        module = load_module()
        recipe_b = module.RECIPES["SFT-B"]
        self.assertEqual(recipe_b.mask_weight, 1.0)
        self.assertEqual(recipe_b.boundary_weight, 1.5)
        self.assertEqual(recipe_b.affected_weight, 0.75)
        self.assertEqual(recipe_b.outside_weight, 0.15)
        self.assertEqual(recipe_b.far_outside_weight, 0.02)

    def test_recipe_parser_rejects_unknown_names(self):
        module = load_module()
        with self.assertRaises(ValueError):
            module.parse_recipes("SFT-A,NOT_A_RECIPE")

    def test_gate_requires_region_safety(self):
        module = load_module()
        rows = [
            {
                "delta_full_psnr": 0.1,
                "delta_mask_psnr": 0.1,
                "delta_boundary_psnr": -0.1,
                "delta_outside_psnr": 0.0,
                "delta_temporal_diff_mae": 0.0,
                "step0_stepn_full_mae": 1.0,
                "step0_stepn_mask_mae": 1.0,
                "step0_stepn_outside_mae": 0.0,
            }
        ]
        status, _ = module.gate30_status(rows)
        self.assertEqual(status, "PSNR_GAIN_WITH_BOUNDARY_OR_OUTSIDE_COST")


if __name__ == "__main__":
    unittest.main()
