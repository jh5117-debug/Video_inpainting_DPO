# Exp25 Gate32 Final Dense Review Readback

Status: `GATE32_FINAL_DENSE_REVIEW_READBACK_COMPLETE`

## Git

- Branch: `research/exp25-vor-or-preference-data`
- HEAD at readback: `d20305fb50b9107b6882f57128a80da2a6fcf6a2`
- Previous milestone commit: `d20305f Add Exp25 Gate32 final review readback`

## Files Read

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/47_exp25_vor_or_preference_data.md`
- `experiment_registry/exp25_vor_or_preference_data/status.md`
- `reports/exp25_gate32_yield_review_20260624.md`
- `reports/exp25_gate32_individual_video_reaudit.md`
- `reports/exp25_gate32_individual_video_reaudit.csv`
- `reports/exp25_gate32_final_video_review_readback.md`
- `reports/exp25_diffueraser_or_root_cause_matrix.md`
- `reports/exp25_diffueraser_or_root_cause_matrix.csv`

## Code Read

- `exp25_vor_or_preference_data/scripts/reaudit_gate32_individual_videos.py`
- `exp25_vor_or_preference_data/scripts/analyze_gate32_yield.py`
- `exp25_vor_or_preference_data/scripts/infer_diffueraser_or_exp25.py`
- `exp25_vor_or_preference_data/scripts/materialize_vor_or_inputs.py`
- `exp25_vor_or_preference_data/scripts/vor_archive_utils.py`

## Already Completed and Not Repeated

- Gate32 materialization: `32/32`
- Gate32 canonical DiffuEraser raw OR generation: `32/32`
- Prior yield review: `medium-hard=11`, `trivial-bad=21`, `too-close=0`
- Prior individual frame/crop reaudit: `black_frame_ratio=0.0` for all 32

## Pending

The previous individual reaudit generated per-sample mp4/contact/crop evidence
but only sampled start/middle/end/mask-max/error-max frames. It explicitly
marked every row as `VISUAL_REVIEW_PENDING_HEADLESS_FRAME_AUDIT_ONLY`.

This milestone must generate and review denser temporal evidence:

- 16 uniformly sampled frames
- start / middle / end
- mask-area max frame
- mask-error max frame
- temporal-difference top-3 frames
- object/mask crop
- affected-region crop
- outside crop
- animated GIF fallback

## Banned Repeats

- Do not rerun Gate32 generation.
- Do not expand Gate128.
- Do not start OR-DPO.
- Do not replace samples or change seed.

## Promotion Gate

This milestone can only complete Gate32 dense temporal review fallback. It
cannot mark `DATA_READY`, `LOSER_UTILITY_PASS`, or `SCIENTIFIC_POSITIVE`.
DiffuEraser root-cause matrix remains required before judging the generator.
