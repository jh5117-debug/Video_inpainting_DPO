import ast
from pathlib import Path
import unittest


class PrepareDataSplitsTest(unittest.TestCase):
    def test_required_keys_are_declared(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "prepare_data_splits.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertIn("REQUIRED_KEYS", names)
        self.assertIn("OPTIONAL_KEYS", names)


if __name__ == "__main__":
    unittest.main()
