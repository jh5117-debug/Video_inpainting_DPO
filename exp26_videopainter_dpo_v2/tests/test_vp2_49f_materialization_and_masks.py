import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(rel: str, name: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class VideoPainterFormal49FTest(unittest.TestCase):
    def test_official_indices_reject_short_video(self):
        mat = load_module("code/materialize_vp2_49f_sources.py", "vp2_mat_test_short")
        with self.assertRaises(ValueError):
            mat.official_indices(frame_count=16, num_frames=49, stride=1, offset=0)

    def test_official_indices_are_exactly_49_unique(self):
        mat = load_module("code/materialize_vp2_49f_sources.py", "vp2_mat_test_unique")
        indices = mat.official_indices(frame_count=60, num_frames=49, stride=1, offset=0)
        self.assertEqual(len(indices), 49)
        self.assertEqual(len(set(indices)), 49)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 48)

    def test_decode_indices_writes_49_unique_frames(self):
        mat = load_module("code/materialize_vp2_49f_sources.py", "vp2_mat_test_decode")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video = tmp_path / "source.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 24, (32, 24))
            for idx in range(60):
                frame = np.zeros((24, 32, 3), dtype=np.uint8)
                frame[:, :, 0] = (idx * 3) % 255
                frame[:, :, 1] = (idx * 7) % 255
                frame[:, :, 2] = (idx * 13) % 255
                cv2.putText(frame, str(idx), (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                writer.write(frame)
            writer.release()
            paths, hashes = mat.decode_indices(video, list(range(49)), tmp_path / "frames")
            self.assertEqual(len(paths), 49)
            self.assertEqual(len(set(hashes)), 49)

    def test_moving_mask_first_frame_gt_and_not_constant(self):
        masks_mod = load_module("code/generate_vp2_moving_br_masks.py", "vp2_mask_test")
        masks, meta = masks_mod.moving_mask_sequence(
            sample_id="sample_a",
            num_frames=49,
            height=64,
            width=96,
            seed=123,
            first_frame_gt=True,
        )
        self.assertEqual(len(masks), 49)
        self.assertEqual(int(masks[0].sum()), 0)
        self.assertGreater(meta["area_mean"], 0.0)
        self.assertGreater(meta["centroid_motion_px"], 0.0)
        later_sums = {int(m.sum()) for m in masks[1:]}
        self.assertGreater(len(later_sums), 1)

    def test_gate64_profiles_are_not_all_ellipse(self):
        masks_mod = load_module("code/generate_vp2_moving_br_masks.py", "vp2_mask_test_profiles")
        profiles = [
            "irregular_freeform",
            "object_like_polygon",
            "soft_blob",
            "edge_touch_freeform",
            "ellipse_circle_subset",
            "thin_structure_freeform",
        ]
        compactness = {}
        for profile in profiles:
            masks, meta = masks_mod.moving_mask_sequence(
                sample_id=f"sample_{profile}",
                num_frames=49,
                height=96,
                width=128,
                seed=456,
                first_frame_gt=True,
                mask_profile=profile,
                area_bucket="medium",
                motion_bucket="medium",
                deformation_bucket="moderate",
                edge_touch_target=profile == "edge_touch_freeform",
            )
            self.assertEqual(meta["mask_profile"], profile)
            self.assertEqual(int(masks[0].sum()), 0)
            self.assertGreater(meta["area_mean"], 0.05)
            compactness[profile] = round(float(meta["compactness_mean"]), 3)
        self.assertGreater(len(set(compactness.values())), 2)

    def test_edge_touch_profile_touches_edge(self):
        masks_mod = load_module("code/generate_vp2_moving_br_masks.py", "vp2_mask_test_edge")
        _, meta = masks_mod.moving_mask_sequence(
            sample_id="sample_edge",
            num_frames=49,
            height=96,
            width=128,
            seed=789,
            first_frame_gt=True,
            mask_profile="edge_touch_freeform",
            area_bucket="medium",
            motion_bucket="low",
            deformation_bucket="slow",
            edge_touch_target=True,
        )
        self.assertGreater(meta["edge_touch_frames"], 0)


if __name__ == "__main__":
    unittest.main()
