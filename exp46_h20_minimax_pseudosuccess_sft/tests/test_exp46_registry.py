from pathlib import Path
import unittest


class Exp46RegistryTest(unittest.TestCase):
    def test_exp46_registry_exists(self):
        root = Path(__file__).resolve().parents[2]
        status_path = root / "experiment_registry" / "exp46_h20_minimax_pseudosuccess_sft" / "status.md"
        self.assertTrue(status_path.exists())


if __name__ == "__main__":
    unittest.main()
