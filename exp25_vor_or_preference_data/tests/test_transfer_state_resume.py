import csv
import tempfile
import unittest
from pathlib import Path

from exp25_vor_or_preference_data.scripts.transfer_effecterase_hal_to_pai import load_completed, MANIFEST_FIELDS


class TransferStateResumeTest(unittest.TestCase):
    def test_load_completed_verified_only(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "manifest.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerow({"filename": "a", "status": "VERIFIED", "size": "10", "hal_sha256": "x", "pai_sha256": "x"})
                writer.writerow({"filename": "b", "status": "FAILED", "size": "20"})
            completed = load_completed(path)
            self.assertIn("a", completed)
            self.assertNotIn("b", completed)


if __name__ == "__main__":
    unittest.main()

