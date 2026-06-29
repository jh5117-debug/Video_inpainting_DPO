import pathlib
import unittest


class Exp45BoundaryTest(unittest.TestCase):
    def test_registry_declares_no_training(self):
        root = pathlib.Path(__file__).resolve().parents[2]
        config = root / "experiment_registry" / "exp45_pai_minimax_pair_scaleup" / "config.yaml"
        text = config.read_text()
        self.assertIn("no_h20_execution: true", text)
        self.assertIn("no_training: true", text)
        self.assertIn("no_optimizer_step: true", text)


if __name__ == "__main__":
    unittest.main()
