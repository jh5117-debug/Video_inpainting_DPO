#!/usr/bin/env python
"""Build labeled side-by-side videos from VBench-standard named folders."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


def load_prompts(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def safe_stem(text: str, max_len: int = 96) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_.")
    return text[:max_len] or "prompt"


def find_named_video(folder: Path, prompt: str, sample_index: int) -> Path:
    stem = f"{prompt}-{sample_index}"
    for suffix in (".mp4", ".gif"):
        path = folder / f"{stem}{suffix}"
        if path.exists():
            return path
    matches = sorted(folder.glob(f"{prompt}-{sample_index}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing video for prompt={prompt!r} sample={sample_index} under {folder}")


def draw_label(frame: np.ndarray, label: str) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 44), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(
        frame,
        label,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.86,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_prompt(canvas: np.ndarray, prompt: str) -> np.ndarray:
    prompt_bar = np.zeros((52, canvas.shape[1], 3), dtype=np.uint8)
    text = prompt if len(prompt) <= 140 else prompt[:137] + "..."
    cv2.putText(
        prompt_bar,
        text,
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([prompt_bar, canvas])


def combine_pair(
    left_path: Path,
    right_path: Path,
    output_path: Path,
    left_label: str,
    right_label: str,
    prompt: str,
    fps: float | None,
) -> None:
    left = cv2.VideoCapture(str(left_path))
    right = cv2.VideoCapture(str(right_path))
    if not left.isOpened():
        raise RuntimeError(f"Could not open {left_path}")
    if not right.isOpened():
        raise RuntimeError(f"Could not open {right_path}")

    left_fps = left.get(cv2.CAP_PROP_FPS) or 10.0
    right_fps = right.get(cv2.CAP_PROP_FPS) or left_fps
    out_fps = float(fps or left_fps or right_fps or 10.0)
    left_w = int(left.get(cv2.CAP_PROP_FRAME_WIDTH))
    left_h = int(left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    right_w = int(right.get(cv2.CAP_PROP_FRAME_WIDTH))
    right_h = int(right.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height = min(left_h, right_h)
    left_width = max(1, round(left_w * height / max(1, left_h)))
    right_width = max(1, round(right_w * height / max(1, right_h)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (left_width + right_width, height + 52),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {output_path}")

    try:
        while True:
            ok_l, frame_l = left.read()
            ok_r, frame_r = right.read()
            if not ok_l or not ok_r:
                break
            frame_l = cv2.resize(frame_l, (left_width, height), interpolation=cv2.INTER_AREA)
            frame_r = cv2.resize(frame_r, (right_width, height), interpolation=cv2.INTER_AREA)
            draw_label(frame_l, left_label)
            draw_label(frame_r, right_label)
            writer.write(draw_prompt(np.hstack([frame_l, frame_r]), prompt))
    finally:
        writer.release()
        left.release()
        right.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_dir", required=True, type=Path)
    parser.add_argument("--right_dir", required=True, type=Path)
    parser.add_argument("--prompts_file", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--left_label", default="baseline")
    parser.add_argument("--right_label", default="dpo")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file)
    written = 0
    missing: list[str] = []
    for idx, prompt in enumerate(prompts):
        try:
            left = find_named_video(args.left_dir, prompt, args.sample_index)
            right = find_named_video(args.right_dir, prompt, args.sample_index)
        except FileNotFoundError as exc:
            missing.append(str(exc))
            continue
        output = args.output_dir / f"{idx + 1:04d}_{safe_stem(prompt)}.mp4"
        combine_pair(left, right, output, args.left_label, args.right_label, prompt, args.fps)
        written += 1

    print(
        f"[named-compare] left={args.left_dir} right={args.right_dir} "
        f"written={written} missing={len(missing)} output={args.output_dir}"
    )
    if missing:
        for item in missing[:20]:
            print(f"[named-compare][missing] {item}")
        if args.strict:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
