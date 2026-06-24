# Exp26 Gate16 Final Video Review

Status: `GATE16_PASSED_WITH_REJECTION`

Date: 2026-06-24

## Scope

This milestone reviews the existing Gate16 official VideoPainter outputs only.
No sample was replaced, no Gate64 was launched, and no DPO training was started.

Input run:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2/exp26_gate16_official_inference`

Final review evidence:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_26_27_followup_20260624/exp26_gate16_final_video_review`

## Review Method

Interactive mp4 playback is not available inside the current Codex UI, so the
review used the fallback allowed by the current instruction:

- opened the per-sample contact sheet for all 16 samples;
- generated and opened per-sample dense temporal packs with 16 uniformly sampled
  frame indices:
  `0, 3, 6, 9, 12, 16, 19, 22, 25, 28, 32, 35, 38, 41, 45, 48`;
- checked start/middle/end coverage, temporal consistency, frame order, visible
  local artifacts, black-frame collapse, and region localization.

All 16 rows have `reviewer_pass=true` in
`reports/exp26_gate16_final_video_review.csv`.

## Classification

| class | count |
| --- | ---: |
| medium-hard | 10 |
| hard-plausible | 5 |
| trivial-bad | 1 |
| technical-invalid | 0 |

Failed true-model row retained:

- `vp2_gate16_BLENDER_CON001_00742`
- class: `trivial-bad`
- reason: persistent large mismatched pale region under the mask.
- decision: true model failure, not wrapper/preprocessing/frame-count failure.

## Pre-registered Gate Check

| criterion | value | status |
| --- | ---: | --- |
| technical valid >= 15/16 | 16/16 | pass |
| systematic failure = 0 | 0 | pass |
| trivial bad <= 2/16 | 1/16 | pass |
| medium-hard/hard-plausible >= 8/16 | 15/16 | pass |
| visual review complete | 16/16 dense packs opened | pass |

## Decision

`GATE16_PASSED_WITH_REJECTION`

Gate16 is now valid for the next Gate64 construction milestone, but this does
not imply `TRAINING_PASS` or `SCIENTIFIC_POSITIVE`. Gate64 still requires a new
readback milestone, mixed-mask construction, full 64-row generation, and full
video review before any VideoPainter DPO micro-training.
