import unittest

from exp25_vor_or_preference_data.scripts.effecterase_common import continuity_report, required_inventory


class InventoryContinuityTest(unittest.TestCase):
    def test_numeric_parts_contiguous(self):
        files = required_inventory([
            {"filename": "README.md", "size": 1},
            {"filename": "VOR-Eval.tar.gz.part_1", "size": 1},
            {"filename": "VOR-Eval.tar.gz.part_2", "size": 1},
            {"filename": "VOR-Train-MASK.tar.gz.part_1", "size": 1},
            {"filename": "VOR-Train.tar.gz.part_1", "size": 1},
        ])
        report = continuity_report(files)
        self.assertTrue(report["VOR-Eval"]["contiguous"])
        self.assertEqual(report["VOR-Eval"]["missing"], [])

    def test_numeric_parts_detect_missing(self):
        files = required_inventory([
            {"filename": "VOR-Eval.tar.gz.part_1", "size": 1},
            {"filename": "VOR-Eval.tar.gz.part_3", "size": 1},
        ])
        report = continuity_report(files)
        self.assertFalse(report["VOR-Eval"]["contiguous"])
        self.assertEqual(report["VOR-Eval"]["missing"], [2])


if __name__ == "__main__":
    unittest.main()

