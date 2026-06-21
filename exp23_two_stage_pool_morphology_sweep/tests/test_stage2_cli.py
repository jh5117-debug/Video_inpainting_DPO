import ast
from pathlib import Path


def _stage2_parser_accepts_aggregation() -> bool:
    source = Path("exp23_two_stage_pool_morphology_sweep/code/train_exp23_stage2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "add_argument":
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--aggregation":
                    return True
    return False


def test_stage2_accepts_runner_aggregation_argument():
    assert _stage2_parser_accepts_aggregation()
