#!/usr/bin/env python3
"""Build Exp38 LocalDPO-v2 corruption preferences.

This is an Exp38-isolated wrapper around the Exp37 pool builder. It tightens the
corruption profiles and outside-preservation gate without modifying Exp37
outputs or source files.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from exp37_minimax_localdpo_badnoise_rescue.scripts import build_localdpo_corruption_pool as base


def profile_region_v2(
    profile: str,
    mask: np.ndarray,
    affected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    boundary = np.clip(base.dilate(mask, 7) - base.erode(mask, 5), 0.0, 1.0)
    affected_soft = base.soft_blur((affected > 0.08).astype(np.float32), 17)
    if profile == "V2_object_micro":
        region = base.soft_blur(mask, 9)
        return region, boundary, 0.14, 1.5
    if profile == "V2_object_mild":
        region = base.soft_blur(mask, 11)
        return region, boundary, 0.20, 2.2
    if profile == "V2_boundary_effect_mild":
        region = np.clip(np.maximum(base.soft_blur(boundary, 9), affected_soft * 0.35), 0.0, 1.0)
        return region, boundary, 0.18, 2.0
    if profile == "V2_object_affected_local":
        region = np.clip(np.maximum(base.soft_blur(mask, 11), affected_soft * 0.45), 0.0, 1.0)
        return region, boundary, 0.22, 2.4
    raise ValueError(profile)


def classify_v2(metrics: dict[str, float]) -> tuple[str, str]:
    outside_ok = metrics["outside_psnr"] >= 50.0 and metrics["outside_mae"] <= 0.35
    if not outside_ok:
        return "TRIVIAL_BAD", "outside damage exceeds Exp38-v2 strict local bound"
    if metrics["mask_mae"] < 1.5 or metrics["mask_psnr"] > 42.0:
        return "TOO_CLOSE", "Exp38-v2 corruption is too close to winner"
    if metrics["mask_psnr"] < 18.0 or metrics["mask_mae"] > 45.0:
        return "TRIVIAL_BAD", "Exp38-v2 corruption is too severe"
    if metrics["mask_psnr"] < 24.0:
        return "HARD_BUT_PLAUSIBLE", "strong but local Exp38-v2 corruption"
    return "MEDIUM_HARD_ELIGIBLE", "clean local Exp38-v2 corruption"


def arg_value(name: str) -> Path:
    idx = sys.argv.index(name)
    return Path(sys.argv[idx + 1])


def copy_report(reports_root: Path, old: str, new: str) -> None:
    src = reports_root / old
    if src.exists():
        shutil.copy2(src, reports_root / new)


def main() -> None:
    base.PROFILE_PAIRS = (
        ("V2_object_mild", "V2_boundary_effect_mild"),
        ("V2_object_micro", "V2_object_mild"),
        ("V2_boundary_effect_mild", "V2_object_affected_local"),
        ("V2_object_affected_local", "V2_object_mild"),
    )
    base.profile_region = profile_region_v2
    base.classify = classify_v2
    base.main()

    reports_root = arg_value("--reports-root")
    copy_report(reports_root, "exp37_localdpo_style_or_corruption_pool.md", "exp38_localdpo_v2_pool.md")
    copy_report(reports_root, "exp37_localdpo_style_or_corruption_pool.csv", "exp38_localdpo_v2_pool.csv")
    copy_report(reports_root, "exp37_localdpo_style_visual_review.csv", "exp38_localdpo_v2_visual_review.csv")
    copy_report(
        reports_root,
        "exp37_localdpo_style_or_corruption_pool_summary.json",
        "exp38_localdpo_v2_pool_summary.json",
    )


if __name__ == "__main__":
    main()
