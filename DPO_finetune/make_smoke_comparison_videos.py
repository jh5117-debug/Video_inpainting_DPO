#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build five-panel smoke comparison videos from one or more smoke roots.

Each output video is:
  GT+mask | ProPainter | COCOCO | DiffuEraser | MiniMax

The script can merge successful candidates across several smoke directories,
which is useful when the four adapters were validated in separate smoke runs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_METHODS = ("propainter", "cococo", "diffueraser", "minimax")


@dataclass
class SampleBundle:
    name: str
    gt_dir: Optional[Path] = None
    mask_dir: Optional[Path] = None
    methods: Dict[str, Path] = field(default_factory=dict)
    source_roots: Dict[str, str] = field(default_factory=dict)


def image_files(path: Path) -> List[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def read_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError(f"failed to read image: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"failed to read mask: {path}")
    return np.where(arr > 0, 255, 0).astype(np.uint8)


def resize_rgb(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    if arr.shape[:2] == (height, width):
        return arr
    return cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape[:2] == (height, width):
        return mask
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def annotate_mask(panel: np.ndarray, mask: np.ndarray, fill: bool) -> np.ndarray:
    out = panel.copy()
    h, w = out.shape[:2]
    mask = resize_mask(mask, w, h)
    active = mask > 0
    if fill and np.any(active):
        red = np.zeros_like(out)
        red[:, :, 0] = 255
        out = np.where(active[:, :, None], (0.55 * out + 0.45 * red).astype(np.uint8), out)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, (255, 40, 40), 3, lineType=cv2.LINE_AA)
    return out


def draw_label(panel: np.ndarray, label: str) -> np.ndarray:
    out = panel.copy()
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(bgr, (8, 8), (tw + 24, th + 24), (0, 0, 0), -1)
    cv2.putText(bgr, label, (16, th + 14), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def candidate_frame_dir(method_root: Path, preferred: Sequence[str]) -> Optional[Path]:
    for name in preferred:
        path = method_root / name
        if image_files(path):
            return path
    return None


def find_sample_dirs(root: Path) -> Iterable[Path]:
    for gt_dir in root.glob("**/gt_frames"):
        sample_dir = gt_dir.parent
        if (sample_dir / "masks").is_dir() and (sample_dir / "candidates").is_dir():
            yield sample_dir


def collect_samples(roots: Sequence[Path], methods: Sequence[str], preferred_dirs: Sequence[str]) -> Dict[str, SampleBundle]:
    samples: Dict[str, SampleBundle] = {}
    for root in roots:
        if not root.is_dir():
            print(f"[warn] smoke root missing: {root}")
            continue
        for sample_dir in sorted(find_sample_dirs(root)):
            sample = samples.setdefault(sample_dir.name, SampleBundle(name=sample_dir.name))
            if image_files(sample_dir / "gt_frames"):
                sample.gt_dir = sample_dir / "gt_frames"
                sample.source_roots["gt"] = str(root)
            if image_files(sample_dir / "masks"):
                sample.mask_dir = sample_dir / "masks"
                sample.source_roots["mask"] = str(root)
            for method in methods:
                frame_dir = candidate_frame_dir(sample_dir / "candidates" / method, preferred_dirs)
                if frame_dir is not None:
                    sample.methods[method] = frame_dir
                    sample.source_roots[method] = str(root)
    return samples


def write_comparison_video(
    sample: SampleBundle,
    methods: Sequence[str],
    out_path: Path,
    fps: int,
    panel_width: Optional[int],
    panel_height: Optional[int],
    max_frames: int,
) -> Dict[str, object]:
    if sample.gt_dir is None or sample.mask_dir is None:
        raise RuntimeError(f"{sample.name}: missing gt_frames or masks")

    gt_files = image_files(sample.gt_dir)
    mask_files = image_files(sample.mask_dir)
    method_files = {method: image_files(sample.methods[method]) for method in methods}
    n = min([len(gt_files), len(mask_files)] + [len(v) for v in method_files.values()])
    if max_frames > 0:
        n = min(n, max_frames)
    if n <= 0:
        raise RuntimeError(f"{sample.name}: no aligned frames to write")

    first = read_rgb(gt_files[0])
    height, width = first.shape[:2]
    width = panel_width or width
    height = panel_height or height

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * (len(methods) + 1), height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {out_path}")

    labels = ["GT + MASK"] + [m.upper() if m == "cococo" else m.title() for m in methods]
    for idx in range(n):
        gt = resize_rgb(read_rgb(gt_files[idx]), width, height)
        mask = resize_mask(read_mask(mask_files[idx]), width, height)
        panels = [annotate_mask(gt, mask, fill=True)]
        for method in methods:
            panel = resize_rgb(read_rgb(method_files[method][idx]), width, height)
            panels.append(annotate_mask(panel, mask, fill=False))
        labeled = [draw_label(panel, label) for panel, label in zip(panels, labels)]
        frame_rgb = np.concatenate(labeled, axis=1)
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    return {
        "sample": sample.name,
        "output": str(out_path),
        "frames": n,
        "methods": {method: str(sample.methods[method]) for method in methods},
        "gt_dir": str(sample.gt_dir),
        "mask_dir": str(sample.mask_dir),
        "source_roots": sample.source_roots,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create GT+mask+four-model smoke comparison videos.")
    parser.add_argument("--smoke_root", action="append", default=[], help="Smoke root. Can be passed multiple times.")
    parser.add_argument("--smoke_outputs_dir", default="", help="Scan DPO_Multimodel_Smoke_* roots under this directory.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--preferred_candidate_dirs", default="composited,normalized_raw,raw_output")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--panel_width", type=int, default=0)
    parser.add_argument("--panel_height", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--include_incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    preferred_dirs = [x.strip() for x in args.preferred_candidate_dirs.split(",") if x.strip()]

    roots = [Path(x).resolve() for x in args.smoke_root]
    if args.smoke_outputs_dir:
        scan_root = Path(args.smoke_outputs_dir).resolve()
        roots.extend(sorted(p for p in scan_root.glob("DPO_Multimodel_Smoke_*") if p.is_dir()))
    if not roots:
        raise SystemExit("no smoke roots supplied; use --smoke_root or --smoke_outputs_dir")

    output_dir = Path(args.output_dir).resolve()
    samples = collect_samples(roots, methods, preferred_dirs)
    manifest: Dict[str, object] = {
        "output_dir": str(output_dir),
        "smoke_roots": [str(p) for p in roots],
        "methods": methods,
        "videos": [],
        "skipped": [],
    }

    for name in sorted(samples):
        sample = samples[name]
        missing = []
        if sample.gt_dir is None:
            missing.append("gt_frames")
        if sample.mask_dir is None:
            missing.append("masks")
        missing.extend(method for method in methods if method not in sample.methods)
        if missing and not args.include_incomplete:
            print(f"[skip] {name}: missing {missing}")
            manifest["skipped"].append({"sample": name, "missing": missing, "source_roots": sample.source_roots})
            continue
        if missing:
            print(f"[warn] {name}: incomplete sample cannot be written as five-panel video: {missing}")
            manifest["skipped"].append({"sample": name, "missing": missing, "source_roots": sample.source_roots})
            continue

        out_path = output_dir / f"{name}_gt_mask_propainter_cococo_diffueraser_minimax.mp4"
        item = write_comparison_video(
            sample,
            methods,
            out_path,
            args.fps,
            args.panel_width or None,
            args.panel_height or None,
            args.max_frames,
        )
        manifest["videos"].append(item)
        print(f"[ok] {name}: {out_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {len(manifest['videos'])} comparison videos")
    print(f"[done] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
