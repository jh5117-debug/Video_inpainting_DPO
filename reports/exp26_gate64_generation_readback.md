# Exp26 Gate64 Generation Readback

Status: `GATE64_GENERATION_READBACK_COMPLETE`

## Git

- Branch: `research/exp26-videopainter-dpo-v2`
- HEAD at readback: `4e3e9b2f5fb00a38df517e06db62f68bf0855ea4`
- Worktree before implementation: clean
- Previous milestone commit: `4e3e9b2 Build locked VideoPainter mixed-mask Gate64 protocol`

## Files Read

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/48_exp26_videopainter_dpo_v2.md`
- `experiment_registry/exp26_videopainter_dpo_v2/status.md`
- `reports/exp26_gate16_final_video_review.md`
- `reports/exp26_gate16_final_video_review.csv`
- `reports/exp26_gate16_final_video_review_summary.json`
- `reports/exp26_gate16_final_video_review_readback.md`
- `reports/exp26_probe4_mask_visual_audit.md`
- `reports/exp26_official_49f_sampler_parity.md`
- `reports/exp26_gate64_readback.md`
- `reports/exp26_br_mask_source_audit.md`
- `reports/exp26_br_mask_source_audit.csv`
- `reports/exp26_gate64_protocol_summary.json`

## Code Read

- `exp26_videopainter_dpo_v2/code/run_vp2_probe4_official_inference.py`
- `exp26_videopainter_dpo_v2/code/review_vp2_gate_outputs.py`
- `exp26_videopainter_dpo_v2/code/materialize_vp2_49f_sources.py`
- `exp26_videopainter_dpo_v2/code/generate_vp2_moving_br_masks.py`
- `exp26_videopainter_dpo_v2/code/build_vp2_gate64_mixed_mask_protocol.py`
- `exp26_videopainter_dpo_v2/scripts/run_vp2_l0_l4_pai.sh`
- Exp25 VOR archive utilities for split tar streaming and path-safety reference:
  `exp25_vor_or_preference_data/scripts/safe_extract_vor_subset.py`
  and `exp25_vor_or_preference_data/scripts/vor_archive_utils.py`

## Dataset and Checkpoint Identity

- Gate64 source manifest:
  `exp26_videopainter_dpo_v2/manifests/vp2_gate64_source_manifest.jsonl`
- Gate64 manifest SHA256:
  `b904be82d58ab7cd897c6759b7351e262f61397d9f90d84df05ae42300dbffb6`
- Rows / scene groups: `64 / 64`
- Formal frames required: `49`
- First frame GT: `true`
- VideoPainter root default:
  `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter`
- Base model default: `ckpt/CogVideoX-5b-I2V`
- Branch checkpoint default: `ckpt/VideoPainter/checkpoints/branch`

## Already Completed and Not Repeated

- L0-L4 official trainer gates are already passed.
- Probe4 official 49F inference is already passed.
- Gate16 final video review is already passed with retained rejection.
- Gate16 failed row `vp2_gate16_BLENDER_CON001_00742` was not replaced.
- Gate64 source/mask protocol is locked and was not resampled.

## Readback Finding

The Gate64 protocol manifest correctly locks a mixed mask distribution, but the
existing `generate_vp2_moving_br_masks.py` implementation still generated only
moving ellipses. Starting Gate64 with that implementation would violate
`vp2_mixed_br_mask_v1`.

## Required Implementation Before Generation

- Add exact VOR-Train/BG selective extraction for the 64 locked member paths.
- Make Gate64 mask generation consume per-row `mask_profile`, `area_bucket`,
  `motion_bucket`, `deformation_bucket`, and `edge_touch_target`.
- Add formal Gate64 official generation runner that writes raw and diagnostic
  comp outputs, status CSV, summary JSON, hashes, contact sheets, and
  side-by-side mp4 files.
- Add a PAI launcher that uses a versioned code snapshot and does not use GPU7.

## Banned Repeats

- Do not rerun Gate16.
- Do not replace the retained Gate16 failure.
- Do not change the Gate64 locked source manifest.
- Do not train VideoPainter DPO until Gate64 data readiness passes.

## Promotion Gate for This Milestone

Gate64 generation may start only after the mixed-mask bug is fixed, tests pass,
and the code is committed and pushed.
