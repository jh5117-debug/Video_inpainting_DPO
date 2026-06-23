# Exp26 VP2 49F Source Split Status

Date: 2026-06-23

## Goal

Build formal 49-frame source splits for VideoPainter self-loser generation:

- train source: 512
- search-dev: 64
- shadow-dev: 64

Formal mode requires exactly/at least 49 real frames and masks; it must not silently fall back to 13-frame plumbing data.

## Current Result

Initial run against:

`/mnt/workspace/hj/nas_hj/data/external/ytbv_2019_full_resolution/train`

failed because no valid 49-frame candidates were found.

Observed sample checks:

- `003234408d`: 36 image frames and 36 masks.
- `0043f083b5`: 20 image frames and 20 masks.

Full PAI split diagnostic:

- valid candidates: `0`
- failed candidates: `3471`
- required valid candidates: `640`
- max frame count seen: `36`
- max mask count seen: `36`
- failure CSV rows including header: `3472`
- statistics file: `exp26_videopainter_dpo_v2/manifests/vp2_49f_source_split_statistics.json`
- failure file: `exp26_videopainter_dpo_v2/manifests/vp2_49f_source_failures.csv`

The active YouTube-VOS directory appears to be a sparse 5-frame-interval extraction rather than a formal 49-frame source.

## Decision

Gate64 VideoPainter official baseline self-loser generation is blocked until a real 49-frame source is located or prepared. The pipeline will not use 13-frame plumbing data for formal self-loser preference data.

Status: `BLOCKED_INSUFFICIENT_49F_SOURCE`.
