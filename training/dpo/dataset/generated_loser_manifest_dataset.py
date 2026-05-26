"""Generated-loser manifest dataset for DPO experiments 5/6/7/8."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def list_image_frames(path: str | Path) -> list[Path]:
    root = Path(path).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    direct = sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if direct:
        return direct
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


class GeneratedLoserManifestDataset(torch.utils.data.Dataset):
    """Read a repaired generated-loser manifest and emit DiffuEraser DPO pairs.

    The returned tensor contract matches the existing DiffuEraser DPO datasets:
    ``pixel_values_pos``, ``pixel_values_neg``, ``conditioning_pixel_values``,
    ``masks``, and ``input_ids``.
    """

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.manifest_path = Path(getattr(args, "preference_manifest", "") or "").expanduser()
        if not self.manifest_path:
            raise ValueError("GeneratedLoserManifestDataset requires --preference_manifest")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"preference manifest not found: {self.manifest_path}")

        self.rows = self._read_manifest(self.manifest_path)
        if not self.rows:
            raise RuntimeError(f"No usable rows in preference manifest: {self.manifest_path}")

        first = self.rows[0]
        self.nframes = int(getattr(args, "nframes", None) or first.get("canonical_num_frames") or first.get("num_frames") or 16)
        self.height = int(getattr(args, "train_height", None) or first.get("canonical_height") or first.get("height") or getattr(args, "resolution", 320))
        self.width = int(getattr(args, "train_width", None) or first.get("canonical_width") or first.get("width") or getattr(args, "resolution", 512))
        self.train_mask_mode = str(getattr(args, "train_mask_mode", "full")).lower()
        self.mask_from_manifest = str_to_bool(getattr(args, "mask_from_manifest", False))
        self.full_mask_value = float(getattr(args, "videodpo_full_mask_value", 0.0))
        self.max_resample_attempts = int(getattr(args, "max_resample_attempts", 64))

        if self.train_mask_mode not in {"full", "partial"}:
            raise ValueError(f"--train_mask_mode must be full or partial, got {self.train_mask_mode}")
        if self.train_mask_mode == "partial" and not self.mask_from_manifest:
            raise ValueError("partial train_mask_mode requires --mask_from_manifest true")

        self._validate_manifest_identity()
        print(
            "GeneratedLoserManifestDataset: "
            f"rows={len(self.rows)} manifest={self.manifest_path} "
            f"train_mask_mode={self.train_mask_mode} mask_from_manifest={self.mask_from_manifest} "
            f"resolution=[{self.height},{self.width}] nframes={self.nframes}"
        )

    def _read_manifest(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        required = {"win_video_path", "final_loser_video_path", "mask_path"}
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = sorted(k for k in required if not row.get(k))
                if missing:
                    raise ValueError(f"{path}:{line_no} missing required fields: {missing}")
                rows.append(row)
        return rows

    def _validate_manifest_identity(self) -> None:
        bad = []
        for row in self.rows[:200]:
            if row.get("generation_source") not in {None, "diffueraser_only"}:
                bad.append((row.get("sample_id"), row.get("generation_source")))
            if row.get("generation_model") not in {None, "diffueraser"}:
                bad.append((row.get("sample_id"), row.get("generation_model")))
        if bad:
            raise ValueError(f"Manifest is not D2 DiffuEraser-only compatible; examples={bad[:5]}")

    def __len__(self) -> int:
        return len(self.rows)

    def tokenize_captions(self, caption: str) -> torch.Tensor:
        if random.random() < self.args.proportion_empty_prompts:
            caption = ""
        inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids[0]

    def _load_rgb_frames(self, frame_dir: str) -> list[Image.Image]:
        paths = list_image_frames(frame_dir)
        if len(paths) < self.nframes:
            raise ValueError(f"Expected at least {self.nframes} frames under {frame_dir}, found {len(paths)}")
        return [Image.open(path).convert("RGB") for path in paths[: self.nframes]]

    def _load_mask_frames(self, mask_dir: str) -> list[Image.Image]:
        paths = list_image_frames(mask_dir)
        if len(paths) < self.nframes:
            raise ValueError(f"Expected at least {self.nframes} masks under {mask_dir}, found {len(paths)}")
        return [Image.open(path).convert("L") for path in paths[: self.nframes]]

    def _image_tensor(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.width, self.height), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return (tensor - 0.5) * 2.0

    def _mask_array(self, mask: Image.Image) -> np.ndarray:
        mask = mask.resize((self.width, self.height), Image.NEAREST)
        return np.asarray(mask, dtype=np.float32) / 255.0

    def _load_row(self, row: dict[str, Any]) -> dict[str, Any]:
        win_frames = self._load_rgb_frames(str(row["win_video_path"]))
        loser_frames = self._load_rgb_frames(str(row["final_loser_video_path"]))
        mask_frames = self._load_mask_frames(str(row["mask_path"]))

        pos_tensors = []
        neg_tensors = []
        cond_tensors = []
        mask_tensors = []

        for win_img, loser_img, mask_img in zip(win_frames, loser_frames, mask_frames):
            pos_tensors.append(self._image_tensor(win_img))
            neg_tensors.append(self._image_tensor(loser_img))

            if self.train_mask_mode == "full":
                cond_tensors.append(torch.full((3, self.height, self.width), -1.0, dtype=torch.float32))
                mask_tensors.append(torch.full((1, self.height, self.width), self.full_mask_value, dtype=torch.float32))
                continue

            mask_np = self._mask_array(mask_img)
            inside = (mask_np > 0.5).astype(np.float32)
            win_np = np.asarray(win_img.resize((self.width, self.height), Image.BILINEAR), dtype=np.float32)
            masked_win = (win_np * (1.0 - inside[:, :, None])).clip(0, 255).astype(np.uint8)
            cond_tensors.append(self._image_tensor(Image.fromarray(masked_win)))
            # Existing DiffuEraser DPO convention: 0 is unknown/hole, 1 is known.
            mask_tensors.append(torch.from_numpy(1.0 - inside).unsqueeze(0).float())

        prompt = str(row.get("prompt") or "")
        return {
            "pixel_values_pos": torch.stack(pos_tensors),
            "pixel_values_neg": torch.stack(neg_tensors),
            "conditioning_pixel_values": torch.stack(cond_tensors),
            "masks": torch.stack(mask_tensors),
            "input_ids": self.tokenize_captions(prompt),
            "sample_id": row.get("sample_id"),
            "pair_index": row.get("pair_index"),
            "mask_id": row.get("mask_id"),
            "prompt": prompt,
            "manifest_row": row,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_error = None
        for attempt in range(self.max_resample_attempts):
            candidate = (index + attempt) % len(self.rows)
            try:
                return self._load_row(self.rows[candidate])
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"Failed to load generated-loser manifest row near index={index}: {last_error}")
