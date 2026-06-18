#!/usr/bin/env python3
"""Exp19 manifest dataset extension with completed-flow conditioning."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dpo.dataset.generated_loser_manifest_dataset import (  # noqa: E402
    GeneratedLoserManifestDataset,
    list_image_frames,
)


def _read_gray_sequence(path: str | Path, nframes: int, size: tuple[int, int]) -> torch.Tensor:
    files = list_image_frames(path)
    if len(files) < nframes:
        raise ValueError(f"expected at least {nframes} frames under {path}, found {len(files)}")
    out = []
    for file in files[:nframes]:
        img = Image.open(file).convert("L").resize(size, Image.BILINEAR)
        out.append(torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None])
    return torch.stack(out, dim=0)


def _load_flow(path: str | Path) -> torch.Tensor:
    arr = np.load(Path(path)).astype(np.float32)
    if arr.ndim != 4:
        raise ValueError(f"flow must be [T-1,2,H,W], got {arr.shape} from {path}")
    return torch.from_numpy(arr)


def _load_conf(path: str | Path) -> torch.Tensor:
    arr = np.load(Path(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[:, None]
    if arr.ndim != 4:
        raise ValueError(f"confidence must be [T-1,1,H,W], got {arr.shape} from {path}")
    return torch.from_numpy(np.clip(arr, 0.0, 1.0))


class Exp19FlowManifestDataset(GeneratedLoserManifestDataset):
    """Generated-loser DPO dataset plus Exp19 flow cache fields.

    The base dataset returns masks in the existing DiffuEraser convention:
    ``1 = known`` and ``0 = hole``. Exp19 converts this to ``hole = 1`` for
    gate construction.
    """

    required_flow_fields = {
        "completed_forward_flow_path",
        "completed_backward_flow_path",
        "flow_confidence_path",
    }

    def _read_manifest(self, path: Path) -> list[dict[str, Any]]:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = sorted(k for k in {"win_video_path", "final_loser_video_path", "mask_path"} if not row.get(k))
                missing += sorted(k for k in self.required_flow_fields if not row.get(k))
                if missing:
                    raise ValueError(f"{path}:{line_no} missing required fields: {missing}")
                rows.append(row)
        return rows

    def _load_row(self, row: dict[str, Any]) -> dict[str, Any]:
        item = super()._load_row(row)
        flow_f = _load_flow(row["completed_forward_flow_path"])[: self.nframes - 1]
        flow_b = _load_flow(row["completed_backward_flow_path"])[: self.nframes - 1]
        conf = _load_conf(row["flow_confidence_path"])[: self.nframes - 1]

        # Resize to training frame size and scale displacement units.
        pair_h, pair_w = flow_f.shape[-2:]
        if (pair_h, pair_w) != (self.height, self.width):
            flat_f = flow_f.reshape(-1, 2, pair_h, pair_w)
            flat_b = flow_b.reshape(-1, 2, pair_h, pair_w)
            scale_x = float(self.width) / max(float(pair_w), 1.0)
            scale_y = float(self.height) / max(float(pair_h), 1.0)
            flat_f = F.interpolate(flat_f, size=(self.height, self.width), mode="bilinear", align_corners=False)
            flat_b = F.interpolate(flat_b, size=(self.height, self.width), mode="bilinear", align_corners=False)
            flat_f[:, 0] *= scale_x
            flat_b[:, 0] *= scale_x
            flat_f[:, 1] *= scale_y
            flat_b[:, 1] *= scale_y
            flow_f = flat_f.reshape(-1, 2, self.height, self.width)
            flow_b = flat_b.reshape(-1, 2, self.height, self.width)
            conf = F.interpolate(conf, size=(self.height, self.width), mode="bilinear", align_corners=False)

        if flow_f.shape[0] < self.nframes - 1:
            raise ValueError(f"flow cache has too few pairs for sample {row.get('sample_id')}: {flow_f.shape[0]}")
        zero_flow = torch.zeros(1, 2, self.height, self.width, dtype=flow_f.dtype)
        zero_conf = torch.zeros(1, 1, self.height, self.width, dtype=conf.dtype)
        flow_f_frame = torch.cat([flow_f, zero_flow], dim=0)
        flow_b_frame = torch.cat([zero_flow, flow_b], dim=0)
        conf_frame = torch.cat([conf, zero_conf], dim=0).clamp(0.0, 1.0)

        known_mask = item["masks"].float()
        hole_mask = (1.0 - known_mask).clamp(0.0, 1.0)
        boundary = self._outer_boundary(hole_mask)
        gate_mask = torch.clamp(hole_mask + 0.75 * boundary, 0.0, 1.0)
        item.update(
            {
                "flow_forward": flow_f_frame,
                "flow_backward": flow_b_frame,
                "flow_confidence": conf_frame,
                "hole_mask": hole_mask,
                "outer_boundary": boundary,
                "flow_gate_mask": gate_mask,
                "flow_cache_row": row,
            }
        )
        return item

    @staticmethod
    def _outer_boundary(hole_mask: torch.Tensor, pixels: int = 1) -> torch.Tensor:
        # hole_mask: [T,1,H,W]
        kernel = pixels * 2 + 1
        dil = F.max_pool2d(hole_mask, kernel_size=kernel, stride=1, padding=pixels)
        return (dil - hole_mask).clamp(0.0, 1.0)


def build_flow_condition(batch: dict[str, torch.Tensor], variant: str) -> tuple[torch.Tensor, dict[str, float]]:
    """Create ``[B,T,7,H,W]`` flow-conditioning tensor."""
    f = batch["flow_forward"].float()
    b = batch["flow_backward"].float()
    conf = batch["flow_confidence"].float()
    hole = batch["hole_mask"].float()
    boundary = batch["outer_boundary"].float()
    if variant == "global":
        gate = conf
    elif variant in {"boundary", "boundary_warp"}:
        gate = conf * batch["flow_gate_mask"].float()
    else:
        raise ValueError(f"Unknown Exp19 variant gate: {variant}")
    cond = torch.cat([f * gate, b * gate, conf, hole, boundary], dim=2)
    finite_conf = conf.detach()
    finite_gate = gate.detach()
    stats = {
        "gate_mean": float(finite_gate.mean().cpu()),
        "gate_p10": float(torch.quantile(finite_gate.flatten(), 0.10).cpu()),
        "gate_p50": float(torch.quantile(finite_gate.flatten(), 0.50).cpu()),
        "gate_p90": float(torch.quantile(finite_gate.flatten(), 0.90).cpu()),
        "flow_conf_mean": float(finite_conf.mean().cpu()),
        "valid_flow_ratio": float((finite_conf > 0.05).float().mean().cpu()),
        "mean_flow_magnitude": float(torch.sqrt((f.pow(2).sum(dim=2) + b.pow(2).sum(dim=2)) * 0.5).mean().cpu()),
    }
    return cond, stats
