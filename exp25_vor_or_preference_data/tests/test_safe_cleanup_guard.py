import tempfile
import unittest
from pathlib import Path

from exp25_vor_or_preference_data.scripts.effecterase_common import safe_remove_tree


class SafeCleanupGuardTest(unittest.TestCase):
    def test_removes_only_under_allowed_parent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            jobs = root / "jobs"
            target = jobs / "file_job"
            target.mkdir(parents=True)
            (target / "x").write_text("data")
            removed = safe_remove_tree(target, jobs)
            self.assertGreater(removed, 0)
            self.assertFalse(target.exists())

    def test_refuses_outside_parent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            allowed = root / "jobs"
            outside = root / "outside"
            allowed.mkdir()
            outside.mkdir()
            with self.assertRaises(RuntimeError):
                safe_remove_tree(outside, allowed)


if __name__ == "__main__":
    unittest.main()

