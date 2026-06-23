#!/usr/bin/env python3
"""Frame-index sampling for DiffuEraser OR clips."""

from __future__ import annotations


def sample_aligned_indices(num_frames: int, target_frames: int = 16) -> list[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    if num_frames == 1:
        return [0] * target_frames
    if num_frames <= target_frames:
        return list(range(num_frames)) + [num_frames - 1] * (target_frames - num_frames)
    return [round(i * (num_frames - 1) / (target_frames - 1)) for i in range(target_frames)]
