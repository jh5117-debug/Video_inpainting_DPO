import ast
from pathlib import Path
import unittest


class Exp43RunnerScaffoldTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[2]
        self.exp_dir = self.root / "exp43_h20_minimax_stage2_sft_runner"

    def test_required_files_exist(self):
        for rel in (
            "precision_policy.py",
            "runner_stage2_sft_ladder.py",
            "launch_single_gpu_preflight.sh",
            "launch_ddp_bf16_safe.sh",
            "configs/bf16_safe_preflight.yaml",
            "manifests/exp43_preflight_train_h20.jsonl",
        ):
            self.assertTrue((self.exp_dir / rel).exists(), rel)

    def test_runner_has_no_shared_trainer_import(self):
        tree = ast.parse((self.exp_dir / "runner_stage2_sft_ladder.py").read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        forbidden = {"inference.metrics", "shared_trainer", "trainer"}
        self.assertTrue(forbidden.isdisjoint(imports))


if __name__ == "__main__":
    unittest.main()
