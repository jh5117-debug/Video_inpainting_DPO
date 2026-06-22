import tempfile
import unittest
from pathlib import Path

from exp23_two_stage_pool_morphology_sweep.code.exp23_trial_runner import latest_checkpoint, resume_args


class RunnerResumeTest(unittest.TestCase):
    def test_resume_latest_when_checkpoint_exists_without_last_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoint-500").mkdir()
            (root / "checkpoint-1500").mkdir()
            self.assertEqual(latest_checkpoint(root), root / "checkpoint-1500")
            self.assertEqual(resume_args(root), ["--resume_from_checkpoint", "latest"])

    def test_no_resume_when_last_weights_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoint-500").mkdir()
            (root / "last_weights" / "unet_main").mkdir(parents=True)
            (root / "last_weights" / "unet_main" / "config.json").write_text("{}")
            self.assertEqual(resume_args(root), [])


if __name__ == "__main__":
    unittest.main()
