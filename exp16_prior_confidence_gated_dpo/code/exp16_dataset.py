#!/usr/bin/env python3
"""Manifest dataset extension for Exp16.

Exp16 requires a real ProPainter prior path per row. This loader refuses rows
without a prior field so training cannot silently degrade to the old Exp11 loss.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dpo.dataset.generated_loser_manifest_dataset import (  # noqa: E402
    GeneratedLoserManifestDataset,
)


PRIOR_KEYS = (
    "prior_frame_dir",
    "propainter_frame_dir",
    "prior_video_path",
    "propainter_video_path",
    "propainter_mp4",
)


def find_prior_path(row: dict[str, Any]) -> str | None:
    for key in PRIOR_KEYS:
        value = row.get(key)
        if value:
            return str(value)
    return None


def audit_manifest_for_prior(path: str | Path, limit: int = 0) -> dict[str, Any]:
    path = Path(path).expanduser()
    total = 0
    with_prior = 0
    missing_examples: list[str] = []
    if not path.exists():
        return {"manifest": str(path), "exists": False, "total": 0, "with_prior": 0, "missing_prior": 0}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            prior = find_prior_path(row)
            if prior:
                with_prior += 1
            elif len(missing_examples) < 10:
                missing_examples.append(str(row.get("sample_id") or row.get("video_name") or total))
            if limit and total >= limit:
                break
    return {
        "manifest": str(path),
        "exists": True,
        "total": total,
        "with_prior": with_prior,
        "missing_prior": total - with_prior,
        "missing_examples": missing_examples,
    }


class Exp16PriorManifestDataset(GeneratedLoserManifestDataset):
    """Generated-loser DPO dataset with additional ProPainter prior tensor."""

    def _read_manifest(self, path: Path) -> list[dict[str, Any]]:
        rows = super()._read_manifest(path)
        missing = []
        for idx, row in enumerate(rows):
            prior = find_prior_path(row)
            if not prior:
                missing.append((idx, row.get("sample_id")))
                continue
            row["prior_frame_dir"] = prior
        if missing:
            examples = ", ".join(f"{i}:{sid}" for i, sid in missing[:5])
            raise ValueError(
                "Exp16 requires a real ProPainter prior path in every manifest row; "
                f"missing={len(missing)} examples={examples}"
            )
        return rows

    def _load_row(self, row: dict[str, Any]) -> dict[str, Any]:
        item = super()._load_row(row)
        prior_frames = self._load_rgb_frames(str(row["prior_frame_dir"]))
        prior_tensors = [self._image_tensor(frame) for frame in prior_frames[: self.nframes]]
        item["prior_pixel_values"] = torch.stack(prior_tensors)
        item["prior_frame_dir"] = row["prior_frame_dir"]
        return item

