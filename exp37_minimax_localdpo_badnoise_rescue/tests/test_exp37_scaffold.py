from pathlib import Path
import unittest


class Exp37ScaffoldTest(unittest.TestCase):
    def test_exp37_scaffold_paths_exist(self):
        root = Path(__file__).resolve().parents[2]
        self.assertTrue((root / "exp37_minimax_localdpo_badnoise_rescue" / "code").is_dir())
        self.assertTrue((root / "exp37_minimax_localdpo_badnoise_rescue" / "scripts").is_dir())
        self.assertTrue(
            (root / "experiment_registry" / "exp37_minimax_localdpo_badnoise_rescue" / "status.md").is_file()
        )
