from pathlib import Path
import unittest


class Exp47RegistryTest(unittest.TestCase):
    def test_exp47_registry_exists(self):
        root = Path(__file__).resolve().parents[2]
        self.assertTrue((root / "experiment_registry/exp47_h20_minimax_pseudosuccess_forensic/status.md").exists())
        self.assertTrue((root / "reports/exp47_pseudosuccess_forensic_readback.md").exists())


if __name__ == "__main__":
    unittest.main()
