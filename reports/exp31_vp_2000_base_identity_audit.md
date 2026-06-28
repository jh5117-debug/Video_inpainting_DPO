# Exp31 VideoPainter 2000 Base Identity Audit

Status: `VIDEOPAINTER_BASE_IDENTITY_AUDIT_PASSED`

- replay_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp31_videopainter_2000step_longrun/exp31_vp2000_base_identity_replay_20260628_091019`
- eval_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp31_videopainter_2000step_longrun/exp31_vp2000_eval_step0_50_2000_20260628_032700`
- GPU: `1` (`CUDA_VISIBLE_DEVICES=1`; GPU0 untouched)
- official/Step0 weight SHA: `5d01728cb0cb605b591f41cbea033db22d5ae72d0b37565957feae71b089be8e` / `5d01728cb0cb605b591f41cbea033db22d5ae72d0b37565957feae71b089be8e`
- Step50 weight SHA: `3849eafbeb9f30a7fb0f52df4c5f0a172d4d437e4161a182a075e15699b2430b`
- Step2000 weight SHA: `fd02a22088da6869fafed437284287b011181882943f81d5ec8b1a493472c148`

## Pass/Block Checks
- architecture_config_same: `True`
- step0_weight_matches_official_base: `True`
- identity_exact: `True`
- replay_exact: `True`
- comp_formula_exact: `True`

## Replay Diff

| comparison | split | sample | stream | frames | resolution | mae | max_abs | hash_equal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replay_official_base_vs_replay_step0 | search | vp2_vor_bg_49f_REAL_ENV242_00112_002_03 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_official_base_vs_replay_step0 | search | vp2_vor_bg_49f_REAL_ENV242_00112_002_03 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_official_base_vs_existing_step0 | search | vp2_vor_bg_49f_REAL_ENV242_00112_002_03 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_official_base_vs_existing_step0 | search | vp2_vor_bg_49f_REAL_ENV242_00112_002_03 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step0_vs_existing_step0 | search | vp2_vor_bg_49f_REAL_ENV242_00112_002_03 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step0_vs_existing_step0 | search | vp2_vor_bg_49f_REAL_ENV242_00112_002_03 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step50_vs_existing_step50 | search | vp2_vor_bg_49f_REAL_ENV242_00112_002_03 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step50_vs_existing_step50 | search | vp2_vor_bg_49f_REAL_ENV242_00112_002_03 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step2000_vs_existing_step2000 | search | vp2_vor_bg_49f_REAL_ENV242_00112_002_03 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step2000_vs_existing_step2000 | search | vp2_vor_bg_49f_REAL_ENV242_00112_002_03 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_official_base_vs_replay_step0 | search | vp2_vor_bg_49f_REAL_ENV800_00003_001_02 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_official_base_vs_replay_step0 | search | vp2_vor_bg_49f_REAL_ENV800_00003_001_02 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_official_base_vs_existing_step0 | search | vp2_vor_bg_49f_REAL_ENV800_00003_001_02 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_official_base_vs_existing_step0 | search | vp2_vor_bg_49f_REAL_ENV800_00003_001_02 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step0_vs_existing_step0 | search | vp2_vor_bg_49f_REAL_ENV800_00003_001_02 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step0_vs_existing_step0 | search | vp2_vor_bg_49f_REAL_ENV800_00003_001_02 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step50_vs_existing_step50 | search | vp2_vor_bg_49f_REAL_ENV800_00003_001_02 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step50_vs_existing_step50 | search | vp2_vor_bg_49f_REAL_ENV800_00003_001_02 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step2000_vs_existing_step2000 | search | vp2_vor_bg_49f_REAL_ENV800_00003_001_02 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step2000_vs_existing_step2000 | search | vp2_vor_bg_49f_REAL_ENV800_00003_001_02 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_official_base_vs_replay_step0 | shadow | vp2_vor_bg_49f_REAL_ENV109_00002_005_02 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_official_base_vs_replay_step0 | shadow | vp2_vor_bg_49f_REAL_ENV109_00002_005_02 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_official_base_vs_existing_step0 | shadow | vp2_vor_bg_49f_REAL_ENV109_00002_005_02 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_official_base_vs_existing_step0 | shadow | vp2_vor_bg_49f_REAL_ENV109_00002_005_02 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step0_vs_existing_step0 | shadow | vp2_vor_bg_49f_REAL_ENV109_00002_005_02 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step0_vs_existing_step0 | shadow | vp2_vor_bg_49f_REAL_ENV109_00002_005_02 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step50_vs_existing_step50 | shadow | vp2_vor_bg_49f_REAL_ENV109_00002_005_02 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step50_vs_existing_step50 | shadow | vp2_vor_bg_49f_REAL_ENV109_00002_005_02 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step2000_vs_existing_step2000 | shadow | vp2_vor_bg_49f_REAL_ENV109_00002_005_02 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step2000_vs_existing_step2000 | shadow | vp2_vor_bg_49f_REAL_ENV109_00002_005_02 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_official_base_vs_replay_step0 | shadow | vp2_vor_bg_49f_BLENDER_FOREST015_00002 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_official_base_vs_replay_step0 | shadow | vp2_vor_bg_49f_BLENDER_FOREST015_00002 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_official_base_vs_existing_step0 | shadow | vp2_vor_bg_49f_BLENDER_FOREST015_00002 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_official_base_vs_existing_step0 | shadow | vp2_vor_bg_49f_BLENDER_FOREST015_00002 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step0_vs_existing_step0 | shadow | vp2_vor_bg_49f_BLENDER_FOREST015_00002 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step0_vs_existing_step0 | shadow | vp2_vor_bg_49f_BLENDER_FOREST015_00002 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step50_vs_existing_step50 | shadow | vp2_vor_bg_49f_BLENDER_FOREST015_00002 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step50_vs_existing_step50 | shadow | vp2_vor_bg_49f_BLENDER_FOREST015_00002 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step2000_vs_existing_step2000 | shadow | vp2_vor_bg_49f_BLENDER_FOREST015_00002 | raw_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |
| replay_step2000_vs_existing_step2000 | shadow | vp2_vor_bg_49f_BLENDER_FOREST015_00002 | comp_frames | 49 | 720x480 vs 720x480 | 0.000000 | 0.0 | `True` |

## Comp/Mask Polarity

The compositor replay recomputed `comp = raw inside mask + winner outside mask` using threshold `mask > 127` and first-frame mask zeroing. Exact zero MAE means the saved comp frames match that formula and polarity for checked samples.

| split | checkpoint | root | sample | frames | first_frame_mask_sum | comp_mae | comp_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| search | official_base | replay | vp2_vor_bg_49f_REAL_ENV242_00112_002_03 | 49 | 0 | 0.000000 | 0.0 |
| search | step0 | replay | vp2_vor_bg_49f_REAL_ENV242_00112_002_03 | 49 | 0 | 0.000000 | 0.0 |
| search | step50 | replay | vp2_vor_bg_49f_REAL_ENV242_00112_002_03 | 49 | 0 | 0.000000 | 0.0 |
| search | step2000 | replay | vp2_vor_bg_49f_REAL_ENV242_00112_002_03 | 49 | 0 | 0.000000 | 0.0 |
| search | step0 | existing | vp2_vor_bg_49f_REAL_ENV242_00112_002_03 | 49 | 0 | 0.000000 | 0.0 |
| search | step50 | existing | vp2_vor_bg_49f_REAL_ENV242_00112_002_03 | 49 | 0 | 0.000000 | 0.0 |
| search | step2000 | existing | vp2_vor_bg_49f_REAL_ENV242_00112_002_03 | 49 | 0 | 0.000000 | 0.0 |
| search | official_base | replay | vp2_vor_bg_49f_REAL_ENV800_00003_001_02 | 49 | 0 | 0.000000 | 0.0 |
| search | step0 | replay | vp2_vor_bg_49f_REAL_ENV800_00003_001_02 | 49 | 0 | 0.000000 | 0.0 |
| search | step50 | replay | vp2_vor_bg_49f_REAL_ENV800_00003_001_02 | 49 | 0 | 0.000000 | 0.0 |
| search | step2000 | replay | vp2_vor_bg_49f_REAL_ENV800_00003_001_02 | 49 | 0 | 0.000000 | 0.0 |
| search | step0 | existing | vp2_vor_bg_49f_REAL_ENV800_00003_001_02 | 49 | 0 | 0.000000 | 0.0 |
| search | step50 | existing | vp2_vor_bg_49f_REAL_ENV800_00003_001_02 | 49 | 0 | 0.000000 | 0.0 |
| search | step2000 | existing | vp2_vor_bg_49f_REAL_ENV800_00003_001_02 | 49 | 0 | 0.000000 | 0.0 |
| shadow | official_base | replay | vp2_vor_bg_49f_REAL_ENV109_00002_005_02 | 49 | 0 | 0.000000 | 0.0 |
| shadow | step0 | replay | vp2_vor_bg_49f_REAL_ENV109_00002_005_02 | 49 | 0 | 0.000000 | 0.0 |
| shadow | step50 | replay | vp2_vor_bg_49f_REAL_ENV109_00002_005_02 | 49 | 0 | 0.000000 | 0.0 |
| shadow | step2000 | replay | vp2_vor_bg_49f_REAL_ENV109_00002_005_02 | 49 | 0 | 0.000000 | 0.0 |
| shadow | step0 | existing | vp2_vor_bg_49f_REAL_ENV109_00002_005_02 | 49 | 0 | 0.000000 | 0.0 |
| shadow | step50 | existing | vp2_vor_bg_49f_REAL_ENV109_00002_005_02 | 49 | 0 | 0.000000 | 0.0 |
| shadow | step2000 | existing | vp2_vor_bg_49f_REAL_ENV109_00002_005_02 | 49 | 0 | 0.000000 | 0.0 |
| shadow | official_base | replay | vp2_vor_bg_49f_BLENDER_FOREST015_00002 | 49 | 0 | 0.000000 | 0.0 |
| shadow | step0 | replay | vp2_vor_bg_49f_BLENDER_FOREST015_00002 | 49 | 0 | 0.000000 | 0.0 |
| shadow | step50 | replay | vp2_vor_bg_49f_BLENDER_FOREST015_00002 | 49 | 0 | 0.000000 | 0.0 |
| shadow | step2000 | replay | vp2_vor_bg_49f_BLENDER_FOREST015_00002 | 49 | 0 | 0.000000 | 0.0 |
| shadow | step0 | existing | vp2_vor_bg_49f_BLENDER_FOREST015_00002 | 49 | 0 | 0.000000 | 0.0 |
| shadow | step50 | existing | vp2_vor_bg_49f_BLENDER_FOREST015_00002 | 49 | 0 | 0.000000 | 0.0 |
| shadow | step2000 | existing | vp2_vor_bg_49f_BLENDER_FOREST015_00002 | 49 | 0 | 0.000000 | 0.0 |

No training, MiniMax, EffectErase adapter, shared trainer, or `inference/metrics.py` changes were made by this audit.
