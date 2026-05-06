"""
Dataset adapter for running open-source VideoDPO on DiffuEraser-style
video-inpainting DPO pairs.

Expected H20 layout:

  DPO_Finetune_Data_Multimodel_v1/
    manifest.json
    <video_name>/
      gt_frames/
      masks/
      neg_frames_1/
      neg_frames_2/
      meta.json

The adapter returns the same batch contract as ``TextVideoDPO``:
``{"video": cat([winner, loser], dim=0), "caption": ..., "dupfactor": 1.0}``,
where each video tensor is ``[C, T, H, W]`` in ``[-1, 1]``.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class VideoInpaintingDPODataset(Dataset):
    def __init__(
        self,
        data_root,
        resolution,
        video_length,
        frame_stride=1,
        subset_split="all",
        clip_length=1.0,
        caption="clean background",
        neg_dirs=None,
        davis_oversample=10,
        chunk_aligned=False,
        seed=20230211,
        dupbeta=0,
        max_resample_attempts=64,
        **_,
    ):
        self.data_root = data_root
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else list(resolution)
        self.video_length = int(video_length)
        self.frame_stride = int(frame_stride)
        self.subset_split = subset_split
        self.clip_length = clip_length
        self.caption = caption
        self.neg_dirs = list(neg_dirs) if neg_dirs else ["neg_frames_1", "neg_frames_2"]
        self.davis_oversample = int(davis_oversample)
        self.chunk_aligned = bool(chunk_aligned)
        self.seed = int(seed)
        self.dupbeta = dupbeta
        self.max_resample_attempts = int(max_resample_attempts)
        self.entries = self._load_entries()

        if not self.entries:
            raise RuntimeError(f"No usable VideoInpainting DPO pairs found from data_root={data_root}")

        print(
            "VideoInpaintingDPODataset: "
            f"entries={len(self.entries)} roots={self._root_summary()} "
            f"resolution={self.resolution} video_length={self.video_length} frame_stride={self.frame_stride}"
        )

    def _root_summary(self) -> str:
        roots = sorted({entry["dataset_root"] for entry in self.entries})
        return ",".join(roots[:3]) + ("..." if len(roots) > 3 else "")

    def _resolve_roots(self) -> list[str]:
        root = Path(self.data_root)
        if root.is_file() and root.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            with root.open("r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)
            roots = cfg.get("META", [])
            return [str(Path(p).expanduser()) for p in roots]
        return [str(root.expanduser())]

    def _load_entries(self) -> list[dict]:
        entries = []
        for root in self._resolve_roots():
            root_entries = self._load_root_entries(root)
            entries.extend(root_entries)
        return entries

    def _load_root_entries(self, root: str) -> list[dict]:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"VideoInpainting DPO root not found: {root}")

        manifest_path = os.path.join(root, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            video_items = self._manifest_video_items(root, manifest)
        else:
            video_items = [(name, name, {}) for name in sorted(os.listdir(root))]

        entries = []
        seen = set()
        for video_name, video_dir_name, info in video_items:
            video_dir = os.path.join(root, video_dir_name)
            if not os.path.isdir(video_dir):
                continue

            gt_dir = os.path.join(video_dir, "gt_frames")
            if not os.path.isdir(gt_dir):
                continue

            gt_files = self._image_files(gt_dir)
            if len(gt_files) < self.video_length:
                continue

            chunks = self._load_chunks(video_dir)
            for neg_name in self._negative_dirs(video_dir):
                neg_dir = os.path.join(video_dir, neg_name)
                neg_files = self._image_files(neg_dir)
                num_frames = min(int(info.get("num_frames", len(gt_files))), len(gt_files), len(neg_files))
                if num_frames < self.video_length:
                    continue

                entry = {
                    "dataset_root": root,
                    "video_name": video_name,
                    "video_dir_name": video_dir_name,
                    "gt_dir": gt_dir,
                    "neg_dir": neg_dir,
                    "neg_id": neg_name,
                    "num_frames": num_frames,
                    "chunks": chunks,
                }
                valid_starts = self._enumerate_valid_starts(entry)
                if not valid_starts:
                    continue
                dedup_key = (video_dir, neg_name)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                entry["valid_starts"] = tuple(valid_starts)
                entries.extend(self._maybe_oversample(entry))

        return entries

    def _manifest_video_items(self, root: str, manifest) -> list[tuple[str, str, dict]]:
        items = []
        if isinstance(manifest, dict):
            iterable = manifest.items()
        elif isinstance(manifest, list):
            iterable = ((str(i), item) for i, item in enumerate(manifest))
        else:
            raise ValueError(f"Unsupported manifest type under {root}: {type(manifest).__name__}")

        for key, info in iterable:
            info = info or {}
            video_dir_name = str(info.get("video_dir") or info.get("video_name") or key)
            if not os.path.isdir(os.path.join(root, video_dir_name)):
                gt_path = str(info.get("gt_frames", ""))
                if gt_path:
                    video_dir_name = os.path.basename(os.path.dirname(gt_path))
            items.append((str(key), video_dir_name, info))
        return items

    def _negative_dirs(self, video_dir: str) -> list[str]:
        names = [name for name in self.neg_dirs if os.path.isdir(os.path.join(video_dir, name))]
        if names:
            return names
        discovered = []
        for name in sorted(os.listdir(video_dir)):
            path = os.path.join(video_dir, name)
            if os.path.isdir(path) and (name.startswith("neg_frames") or "negative" in name or name.startswith("lose")):
                discovered.append(name)
        return discovered

    def _load_chunks(self, video_dir: str):
        meta_path = os.path.join(video_dir, "meta.json")
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, "r") as f:
                return json.load(f).get("chunks")
        except Exception:
            return None

    def _maybe_oversample(self, entry: dict) -> list[dict]:
        repeats = self.davis_oversample if entry["video_name"].startswith("davis_") and self.davis_oversample > 1 else 1
        return [dict(entry, repeat_slot=i, repeats=repeats) for i in range(repeats)]

    def _enumerate_valid_starts(self, entry: dict) -> list[int]:
        span = (self.video_length - 1) * self.frame_stride + 1
        if self.chunk_aligned and entry.get("chunks"):
            starts = []
            for chunk in entry["chunks"]:
                c_start = int(chunk.get("start", 0))
                c_end = int(chunk.get("end", 0))
                if c_end - c_start >= span:
                    starts.extend(range(c_start, c_end - span + 1))
            if starts:
                return sorted(set(starts))
        return list(range(0, max(0, entry["num_frames"] - span) + 1))

    def _image_files(self, frame_dir: str) -> list[str]:
        return sorted(
            os.path.join(frame_dir, name)
            for name in os.listdir(frame_dir)
            if name.lower().endswith(IMAGE_EXTS)
        )

    def _select_start(self, entry: dict, index: int) -> int:
        valid_starts = entry["valid_starts"]
        rng = random.Random(f"{self.seed}:{entry['video_dir_name']}:{entry['neg_id']}:{index}")
        return valid_starts[rng.randrange(len(valid_starts))]

    def _frame_indices(self, entry: dict, index: int) -> list[int]:
        start = self._select_start(entry, index)
        return [start + i * self.frame_stride for i in range(self.video_length)]

    def _read_video(self, frame_dir: str, frame_indices: Iterable[int]) -> torch.Tensor:
        files = self._image_files(frame_dir)
        frames = []
        height, width = int(self.resolution[0]), int(self.resolution[1])
        for idx in frame_indices:
            with Image.open(files[idx]) as img:
                img = img.convert("RGB").resize((width, height), Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32)
            frames.append(torch.from_numpy(arr).permute(2, 0, 1))
        video = torch.stack(frames, dim=1)
        return video.div(255.0).sub(0.5).mul(2.0)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        last_error = None
        for attempt in range(self.max_resample_attempts):
            real_index = (index + attempt) % len(self.entries)
            entry = self.entries[real_index]
            try:
                frame_indices = self._frame_indices(entry, real_index)
                winner = self._read_video(entry["gt_dir"], frame_indices)
                loser = self._read_video(entry["neg_dir"], frame_indices)
                return {
                    "video": torch.cat([winner, loser], dim=0),
                    "caption": self.caption,
                    "dupfactor": 1.0,
                    "video_name": entry["video_name"],
                    "neg_id": entry["neg_id"],
                }
            except (FileNotFoundError, OSError, IndexError, ValueError) as exc:
                last_error = exc
                continue
        raise RuntimeError(f"Failed to load a valid VideoInpainting DPO pair near index={index}: {last_error}")
