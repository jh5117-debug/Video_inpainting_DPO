# Exp25 Gate32 Final Video Review Readback

Date: 2026-06-24

## Branch / HEAD

- Branch: `research/exp25-vor-or-preference-data`
- HEAD: `6b700109da1975cc27268eeb66ba9b1d96002004`
- Remote fetch: completed with `git fetch --all --prune`
- Worktree status before milestone: clean

## Files Read

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/47_exp25_vor_or_preference_data.md`
- `experiment_registry/exp25_vor_or_preference_data/status.md`
- `reports/exp25_gate32_yield_review_20260624.md`
- `reports/exp25_gate32_individual_video_reaudit.md`
- `reports/exp25_diffueraser_or_root_cause_matrix.md`

## Source Of Truth For This Milestone

- Existing Gate32 canonical DiffuEraser raw OR candidates only.
- Generator identity: `diffueraser_or_none_propainter_abd3ad48f60f`.
- Canonical protocol: no PCM, ProPainter prior, mask dilation 0, raw/no-comp,
  24 frames.
- PAI candidate root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/preference_candidates/gate32_canonical_raw6_d0`
- Existing review roots:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2/exp25_gate32_yield_review`
  and
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_26_27_followup_20260624/exp25_gate32_individual_video_reaudit_v2`

## Completed Work Not To Repeat

- Gate32 materialization and DiffuEraser generation completed `32/32`.
- Prior metric/contact-sheet review found `medium-hard=11`,
  `trivial-bad=21`, `too-close=0`.
- Individual frame/crop reaudit found whole-video black-frame collapse is not
  present (`black_frame_ratio=0.0` for all rows).

## Banned Repeats / Banned Promotions

- Do not expand Gate128.
- Do not start OR-DPO.
- Do not replace or resample Gate32 rows.
- Do not mark `DATA_READY`, `LOSER_UTILITY_PASS`, or `SCIENTIFIC_POSITIVE`.

## Promotion Gate

This milestone can only mark Gate32 final visual review complete. It cannot
promote the data pool. The required next scientific step remains the
DiffuEraser OR root-cause matrix.
