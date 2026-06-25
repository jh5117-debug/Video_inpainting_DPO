import unittest

import numpy as np
from PIL import Image

from exp27_paper_grounded_preference_study.code.localdpo_24f_adaptation import (
    P32_GATE,
    classify_pair,
    fallback_bezier_masks,
    select_manifest_rows,
    summarize_gate,
)


class TestLocalDpo24FAdaptation(unittest.TestCase):
    def test_p32_gate_requires_full_video_review(self):
        rows = []
        for i in range(32):
            rows.append(
                {
                    "technical_valid": True,
                    "classification": "MEDIUM_HARD_ELIGIBLE",
                    "global_collapse": False,
                    "outside_preservation_passed": True,
                    "review_assets": {} if i == 0 else {"temporal_strip": f"{i}.jpg"},
                }
            )
        summary = summarize_gate(P32_GATE, rows)
        self.assertEqual(summary["status"], "P32_FAILED_OR_PENDING")
        self.assertEqual(summary["video_review"], 31)

    def test_p32_gate_thresholds(self):
        rows = []
        for i in range(32):
            rows.append(
                {
                    "technical_valid": i < 30,
                    "classification": "MEDIUM_HARD_ELIGIBLE" if i < 24 else "HARD_BUT_PLAUSIBLE",
                    "global_collapse": False,
                    "outside_preservation_passed": True,
                    "review_assets": {"temporal_strip": f"{i}.jpg"},
                }
            )
        summary = summarize_gate(P32_GATE, rows)
        self.assertEqual(summary["status"], "P32_PASSED")
        self.assertEqual(summary["technical_valid"], 30)

    def test_fallback_bezier_mask_geometry_for_unit_tests(self):
        masks = fallback_bezier_masks(seed=7, frames=24, height=48, width=80)
        self.assertEqual(len(masks), 24)
        areas = [np.asarray(m, dtype=np.uint8).mean() for m in masks]
        self.assertGreater(max(areas), 0)
        self.assertGreater(len(set(int(a) for a in areas)), 1)

    def test_select_manifest_rows_requires_winner_and_mask(self):
        rows = [
            {"sample_id": "bad0", "win_video_path": "x"},
            {"sample_id": "bad1", "mask_path": "m"},
            {"sample_id": "ok0", "win_video_path": "x", "mask_path": "m"},
            {"sample_id": "ok1", "win_video_path": "y", "mask_path": "n"},
        ]
        selected = select_manifest_rows(rows, 8, seed=1)
        self.assertEqual([row["sample_id"] for _, row in selected], ["ok0", "ok1"])

    def test_classify_pair_detects_outside_preservation(self):
        clean = [Image.new("RGB", (32, 24), (120, 100, 80)) for _ in range(4)]
        masks = []
        loser = []
        for frame in clean:
            m = Image.new("L", frame.size, 0)
            arr = np.asarray(m, dtype=np.uint8).copy()
            arr[6:16, 8:20] = 255
            masks.append(Image.fromarray(arr, "L"))
            larr = np.asarray(frame, dtype=np.uint8).copy()
            larr[6:16, 8:20] = (80, 80, 80)
            loser.append(Image.fromarray(larr, "RGB"))
        metrics = classify_pair(clean, masks, loser)
        self.assertTrue(metrics["outside_preservation_passed"])
        self.assertTrue(metrics["technical_valid"])
        self.assertIn(metrics["classification"], {"MEDIUM_HARD_ELIGIBLE", "HARD_BUT_PLAUSIBLE"})


if __name__ == "__main__":
    unittest.main()
