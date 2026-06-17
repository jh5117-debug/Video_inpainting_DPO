# PRD 35: OR Visual Green Bug Fix

Date: 2026-06-17

## Problem

HAL visual outputs under:

```text
/home/hj/dpo-2-1-exp/exp15_or_benchmark_davis50_visuals
```

were reported as all green in playback.

## Audit Result

Debug frames showed:

- raw input frames are normal RGB;
- raw method outputs are normal RGB;
- mask overlay is normal;
- old contact sheets are normal;
- old mp4s were encoded with OpenCV `mp4v`, which is not robust across viewers.

The issue is best treated as an mp4 encoding/playback compatibility bug, not a method-output or path-mapping bug.

## Fix

Visual generation now uses Exp15-local static ffmpeg from `imageio-ffmpeg` and encodes:

```text
H.264 / yuv420p / libx264
```

## Fixed Paths

PAI:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp15_or_benchmark_davis50_fixed/visual_grids
```

HAL:

```text
/home/hj/dpo-2-1-exp/exp15_or_benchmark_davis50_visuals_fixed
```

## Status

Fixed visual grids completed: 50 videos + 50 contact sheets.
