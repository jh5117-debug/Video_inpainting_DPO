#!/usr/bin/env python3
"""Manifest dataset extension for Exp18 propagation-gated DPO."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dpo.dataset.generated_loser_manifest_dataset import (  # noqa: E402
    GeneratedLoserManifestDataset,
    list_image_frames,
)


PROP_FRAME_KEYS = (
    "propagated_frame_dir",
    "multiframe_propagated_frame_dir",
    "propagation_frame_dir",
)
CONF_KEYS = (
    "propagation_confidence_dir",
    "prop_confidence_dir",
    "confidence_map_dir",
)
ORACLE_CONF_KEYS = (
    "oracle_confidence_dir",
    "oracle_prop_confidence_dir",
)


def _first_existing_field(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return None


def audit_manifest_for_propagation(path: str | Path, limit: int = 0, confidence_mode: str = "flow_agreement") -> dict[str, Any]:
    path = Path(path).expanduser()
    total = 0
    usable = 0
    missing_examples: list[str] = []
    if not path.exists():
        return {"manifest": str(path), "exists": False, "total": 0, "usable": 0, "missing": 0}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            prop = _first_existing_field(row, PROP_FRAME_KEYS)
            conf_keys = ORACLE_CONF_KEYS if confidence_mode == "oracle" else CONF_KEYS
            conf = _first_existing_field(row, conf_keys)
            if prop and conf:
                usable += 1
            elif len(missing_examples) < 10:
                missing_examples.append(str(row.get("sample_id") or row.get("video_name") or total))
            if limit and total >= limit:
                break
    return {
        "manifest": str(path),
        "exists": True,
        "total": total,
        "usable": usable,
        "missing": total - usable,
        "confidence_mode": confidence_mode,
        "missing_examples": missing_examples,
    }


class Exp18PropagationManifestDataset(GeneratedLoserManifestDataset):
    """Generated-loser DPO dataset with propagation prior and confidence maps."""

    def _read_manifest(self, path: Path) -> list[dict[str, Any]]:
        rows = super()._read_manifest(path)
        confidence_mode = str(getattr(self.args, "confidence_mode", "flow_agreement")).lower()
        conf_keys = ORACLE_CONF_KEYS if confidence_mode == "oracle" else CONF_KEYS
        missing = []
        for idx, row in enumerate(rows):
            prop = _first_existing_field(row, PROP_FRAME_KEYS)
            conf = _first_existing_field(row, conf_keys)
            if not prop or not conf:
                missing.append((idx, row.get("sample_id"), bool(prop), bool(conf)))
                continue
            row["propagated_frame_dir"] = prop
            row["propagation_confidence_dir"] = conf
        if missing:
            examples = ", ".join(f"{i}:{sid}:prop={has_p}:conf={has_c}" for i, sid, has_p, has_c in missing[:5])
            raise ValueError(
                "Exp18 requires real multi-frame propagation frames and confidence maps in every row; "
                f"missing={len(missing)} examples={examples}"
            )
        return rows

    def _load_gray_sequence(self, frame_dir: str) -> list[Image.Image]:
        paths = list_image_frames(frame_dir)
        if len(paths) < self.nframes:
            raise ValueError(f"Expected at least {self.nframes} confidence frames under {frame_dir}, found {len(paths)}")
        return [Image.open(path).convert("L") for path in paths[: self.nframes]]

    def _confidence_tensor(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.width, self.height), Image.NEAREST)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0).float().clamp(0.0, 1.0)

    def _load_row(self, row: dict[str, Any]) -> dict[str, Any]:
        item = super()._load_row(row)
        prop_frames = self._load_rgb_frames(str(row["propagated_frame_dir"]))
        prop_tensors = [self._image_tensor(frame) for frame in prop_frames[: self.nframes]]
        conf_frames = self._load_gray_sequence(str(row["propagation_confidence_dir"]))
        conf_tensors = [self._confidence_tensor(frame) for frame in conf_frames[: self.nframes]]
        item["propagated_pixel_values"] = torch.stack(prop_tensors)
        item["propagation_confidence"] = torch.stack(conf_tensors)
        item["propagated_frame_dir"] = row["propagated_frame_dir"]
        item["propagation_confidence_dir"] = row["propagation_confidence_dir"]
        item["avg_num_sources_used"] = row.get("avg_num_sources_used")
        item["propagated_region_psnr"] = row.get("propagated_region_psnr")
        item["propagation_coverage"] = row.get("propagation_coverage")
        return item

