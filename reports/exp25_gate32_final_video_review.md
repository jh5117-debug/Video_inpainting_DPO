# Exp25 Gate32 Final Dense Temporal Review

Date: 2026-06-24 UTC

## Scope

This milestone reaudits the existing 32 canonical DiffuEraser VOR OR raw
losers. It does not regenerate Gate32, does not expand Gate128, and does not
start OR-DPO.

PAI runtime:

`/home/hj/exp25_gate32_dense_review_runs/gate32_dense_54507fd_20260625_031604`

Tracked outputs:

- `reports/exp25_gate32_final_video_review.csv`
- `reports/exp25_gate32_final_video_review_summary.json`

Large visual artifacts were copied locally for inspection but are not intended
for git:

`reports/exp25_gate32_dense_review_artifacts_20260625/`

## Method

The run used the dense temporal fallback because interactive mp4 playback is
not available in this execution channel. For each sample the script generated:

- one mp4;
- one animated GIF;
- one dense sample contact sheet with temporal frames and task-specific
  evidence columns;
- blind, informed, and reconciled classifications in CSV.

The copied artifact directory contains:

- `32/32` mp4 files;
- `32/32` animated GIFs;
- `32/32` sample contact sheets.

The current fallback did not emit separate object/affected/outside crop files
as standalone artifacts; the contact sheets contain those columns. Therefore
this milestone remains a dense-review fallback, not a full interactive playback
promotion.

## Results

| class | count |
| --- | ---: |
| `MEDIUM_HARD_ELIGIBLE` | 11 |
| `TRIVIAL_BAD` | 21 |
| `TOO_CLOSE` | 0 |
| `TECHNICAL_INVALID` | 0 |

Other checks:

- `reviewer_pass=true`: `32/32`
- `black_frame_ratio=0.0`: `32/32`
- dominant trivial-bad artifact: `large_mask_region_mismatch_in_raw_frames`
- mean mask PSNR for medium-hard rows: `22.7107`
- mean mask PSNR for trivial-bad rows: `14.4211`

Medium-hard samples:

- `BLENDER_CON001_00332`
- `REAL_ENV114_00004_004_02`
- `BLENDER_CON001_00742`
- `BLENDER_CON001_00744`
- `BLENDER_FOREST039_00360`
- `BLENDER_FOREST039_00051`
- `BLENDER_FOREST039_00009`
- `BLENDER_CON001_00590`
- `BLENDER_CON001_00636`
- `BLENDER_CON001_00843`
- `BLENDER_CON001_00756`

## Manual Spot Check

Opened representative dense contact sheets:

- `BLENDER_CON001_00332`: medium-hard; raw loser is not black, outside area is
  mostly preserved, and the mask/affected region shows visible but bounded
  local deviation.
- `BLENDER_FOREST039_00117`: trivial-bad; raw output is not full black, but
  object/mask-region content is semantically mismatched and rough across the
  timeline.
- `REAL_ENV159_00010_003_05`: trivial-bad; the affected/mask region shows
  persistent object/task mismatch rather than a transient display artifact.

## Interpretation

This confirms the earlier yield picture:

- the black/purple concern is not whole-video black-frame collapse;
- the main failure is low OR loser utility in the task/mask region under the
  current canonical DiffuEraser raw6 stack;
- medium-hard yield is `11/32`, too weak for immediate Gate128 expansion;
- no `too-close` rows exist, so seed2 supplementation is not triggered.

Current decision:

`GATE32_FINAL_DENSE_REVIEW_COMPLETE_YIELD_POOR`

Next required Exp25 milestone remains the DiffuEraser OR root-cause matrix.
Do not set `DATA_READY`, `LOSER_UTILITY_PASS`, or `SCIENTIFIC_POSITIVE`.
