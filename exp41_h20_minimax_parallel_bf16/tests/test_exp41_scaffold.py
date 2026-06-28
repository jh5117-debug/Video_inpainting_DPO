from pathlib import Path
import unittest


class Exp41ScaffoldTest(unittest.TestCase):
    def test_exp41_scaffold_files_exist(self):
        root = Path(__file__).resolve().parents[2]
        self.assertTrue((root / "PRD/56_exp41_h20_minimax_parallel_bf16.md").exists())
        self.assertTrue((root / "experiment_registry/exp41_h20_minimax_parallel_bf16/status.md").exists())
        self.assertTrue((root / "reports/exp41_h20_minimax_parallel_readback.md").exists())
