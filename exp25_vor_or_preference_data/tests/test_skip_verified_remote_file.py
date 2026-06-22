import csv
import tempfile
import unittest
from pathlib import Path

from exp25_vor_or_preference_data.scripts.transfer_effecterase_hal_to_pai import load_completed, MANIFEST_FIELDS


class SkipVerifiedRemoteFileTest(unittest.TestCase):
    def test_sha_must_match_for_completed(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "manifest.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerow({"filename": "ok", "status": "VERIFIED", "size": "1", "hal_sha256": "a", "pai_sha256": "a"})
                writer.writerow({"filename": "bad", "status": "VERIFIED", "size": "1", "hal_sha256": "a", "pai_sha256": "b"})
            completed = load_completed(path)
            self.assertEqual(completed["ok"]["hal_sha256"], completed["ok"]["pai_sha256"])
            self.assertNotEqual(completed["bad"]["hal_sha256"], completed["bad"]["pai_sha256"])


if __name__ == "__main__":
    unittest.main()

