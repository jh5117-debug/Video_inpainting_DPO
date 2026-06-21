import ast
import unittest
from pathlib import Path


def _parser_has_argument(source_path: str, option: str) -> bool:
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "add_argument":
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == option:
                    return True
    return False


class Exp23LegacyOuterCliTest(unittest.TestCase):
    def test_stage1_and_stage2_accept_boundary_mode_cli(self):
        self.assertTrue(_parser_has_argument("exp23_two_stage_pool_morphology_sweep/code/train_stage1.py", "--boundary_mode"))
        self.assertTrue(_parser_has_argument("exp23_two_stage_pool_morphology_sweep/code/train_exp23_stage2.py", "--boundary_mode"))

    def test_legacy_builder_no_longer_defaults_to_both(self):
        source = Path("exp23_two_stage_pool_morphology_sweep/code/train_stage1.py").read_text(encoding="utf-8")
        self.assertNotIn('os.environ.get("BOUNDARY_MODE", "both")', source)
        self.assertIn("requires explicit --boundary_mode", source)


if __name__ == "__main__":
    unittest.main()
