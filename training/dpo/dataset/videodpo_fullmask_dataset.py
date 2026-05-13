"""VideoDPO pair dataset adapted to DiffuEraser full-mask conditioning.

This is the first "minimum-variable" bridge experiment:

* data stays VideoDPO-style winner/loser text-to-video pairs;
* task stays text-to-video preference alignment;
* model side becomes DiffuEraser/BrushNet by giving it a full-hole mask and a
  fully masked conditioning image.

DiffuEraser training code expects:
``pixel_values_pos``, ``pixel_values_neg``, ``conditioning_pixel_values``,
``masks`` and ``input_ids``.  For a full-hole generation setting, the masked
image is black everywhere (normalized to -1), and the BrushNet mask follows the
existing training convention: 0 means unknown/hole after the DAVIS mask is
inverted in the original dataset.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

try:
    from decord import VideoReader, cpu
except Exception:  # pragma: no cover - import is environment-dependent
    VideoReader = None
    cpu = None


class VideoDPOFullMaskDiffuEraserDataset(torch.utils.data.Dataset):
    """Read VideoDPO VC2 pairs and emit DiffuEraser DPO training samples."""

    def __init__(self, args, tokenizer, dpo_data_root: Optional[str] = None):
        self.args = args
        self.tokenizer = tokenizer
        self.data_root = dpo_data_root or getattr(args, "dpo_data_root", None)
        if not self.data_root:
            raise ValueError("VideoDPO full-mask dataset requires --dpo_data_root pointing to train_data.yaml")

        self.nframes = int(args.nframes)
        self.resolution = int(args.resolution)
        self.height = int(getattr(args, "train_height", None) or self.resolution)
        self.width = int(getattr(args, "train_width", None) or self.resolution)
        self.frame_stride = int(getattr(args, "videodpo_frame_stride", 1))
        self.clip_length = float(getattr(args, "videodpo_clip_length", 1.0))
        self.base_seed = int(getattr(args, "seed", 0) or 0)
        self.full_mask_value = float(getattr(args, "videodpo_full_mask_value", 0.0))
        self.max_resample_attempts = int(getattr(args, "max_resample_attempts", 64))

        self.video_roots = self._resolve_video_roots(self.data_root)
        self.videos = []
        self.pairs = []
        self._load_all_roots()
        if not self.pairs:
            raise RuntimeError(f"No VideoDPO pairs found from {self.data_root}")
        print(
            "VideoDPOFullMaskDiffuEraserDataset: "
            f"pairs={len(self.pairs)} roots={len(self.video_roots)} "
            f"resolution=[{self.height},{self.width}] nframes={self.nframes} "
            f"frame_stride={self.frame_stride}"
        )

    def _resolve_video_roots(self, data_root: str) -> list[Path]:
        root = Path(data_root).expanduser()
        if root.is_file() and root.suffix.lower() in {".yaml", ".yml"}:
            with root.open("r", encoding="utf-8") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)
            base = root.parent
            roots = []
            for item in cfg.get("META", []):
                path = Path(str(item)).expanduser()
                if not path.is_absolute():
                    candidates = []
                    if os.environ.get("VIDEODPO_DATA_BASE"):
                        candidates.append(Path(os.environ["VIDEODPO_DATA_BASE"]).expanduser() / path)
                    if os.environ.get("VIDEODPO_REPO"):
                        candidates.append(Path(os.environ["VIDEODPO_REPO"]).expanduser() / path)
                    candidates.extend([base / path, Path.cwd() / path])
                    path = next((p for p in candidates if p.exists()), candidates[0]).resolve()
                roots.append(path)
            return roots
        return [root.resolve()]

    def _load_all_roots(self) -> None:
        for root in self.video_roots:
            self._load_root(root)

    def _load_root(self, root: Path) -> None:
        metadata_path = root / "metadata.json"
        pair_path = root / "pair.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found under VideoDPO root: {root}")
        if not pair_path.exists():
            raise FileNotFoundError(f"pair.json not found under VideoDPO root: {root}")

        video_offset = len(self.videos)
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        for item in metadata:
            clip_path = Path(item["basic"]["clip_path"])
            if not clip_path.is_absolute():
                clip_path = root / clip_path
            self.videos.append({
                "clip_path": str(clip_path),
                "caption": self._metadata_caption(item),
                "duration": float(item.get("basic", {}).get("clip_duration", 0.0) or 0.0),
            })

        with pair_path.open("r", encoding="utf-8") as f:
            pairs = json.load(f)
        for item in pairs:
            caption = item.get("frame_caption") or item.get("caption")
            if isinstance(caption, list):
                caption = caption[0] if caption else ""
            self.pairs.append({
                "winner": video_offset + int(item["video1"]),
                "loser": video_offset + int(item["video2"]),
                "caption": str(caption or ""),
            })

    def _metadata_caption(self, item: dict) -> str:
        captions = item.get("misc", {}).get("frame_caption", [])
        if isinstance(captions, list) and captions:
            return str(captions[0])
        if isinstance(captions, str):
            return captions
        return ""

    def __len__(self) -> int:
        return len(self.pairs)

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

    def _read_video(self, path: str, index: int) -> torch.Tensor:
        if VideoReader is None:
            raise ImportError("decord is required for VideoDPO full-mask dataset video loading")
        reader = VideoReader(path, ctx=cpu(0), width=self.width, height=self.height)
        if len(reader) < self.nframes:
            raise ValueError(f"Video too short for nframes={self.nframes}: {path}")
        all_frames = list(range(0, len(reader), self.frame_stride))
        if len(all_frames) < self.nframes:
            all_frames = list(range(0, len(reader), 1))
        start = random.randint(0, len(all_frames) - self.nframes)
        frame_indices = all_frames[start:start + self.nframes]
        frames = reader.get_batch(frame_indices).asnumpy()
        frames = torch.tensor(np.asarray(frames, dtype=np.uint8)).permute(0, 3, 1, 2).float()
        return (frames / 255.0 - 0.5) * 2.0

    def _load_pair(self, pair: dict, index: int) -> dict:
        winner_info = self.videos[pair["winner"]]
        loser_info = self.videos[pair["loser"]]
        caption = pair["caption"] or winner_info["caption"] or loser_info["caption"]

        pos = self._read_video(winner_info["clip_path"], index)
        neg = self._read_video(loser_info["clip_path"], index)
        conditioning = torch.full_like(pos, -1.0)
        masks = torch.full(
            (self.nframes, 1, self.height, self.width),
            self.full_mask_value,
            dtype=pos.dtype,
        )
        return {
            "pixel_values_pos": pos,
            "pixel_values_neg": neg,
            "conditioning_pixel_values": conditioning,
            "masks": masks,
            "input_ids": self.tokenize_captions(caption),
        }

    def __getitem__(self, index: int) -> dict:
        last_error = None
        for attempt in range(self.max_resample_attempts):
            candidate = (index + attempt) % len(self.pairs)
            try:
                return self._load_pair(self.pairs[candidate], candidate)
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"Failed to load VideoDPO full-mask pair near index={index}: {last_error}")
