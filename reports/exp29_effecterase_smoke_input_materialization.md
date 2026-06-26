# Exp29 EffectErase Smoke Input Materialization

Date: 2026-06-26

Status: `EFFECTERASE_SMOKE_INPUTS_BLOCKED`

## Summary

- Manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/effecterase_smoke_preregistered.jsonl`
- Manifest SHA256: `54fd62a97fa69f2f17590488136d426cee77de0ed02548c46a83d8818be2b137`
- Rows: 6
- Ready rows: 5
- Diagnostic FPS used for materialized mp4s: 8
- Fixed resolution: 832x480
- Fixed frame count: 17
- VOR-Eval use: false
- Training eligibility: false for all rows
- Scientific role: diagnostic_only_vor_confounded for all rows

## Per-Row Result

| sample_id | source_type | mask_bucket | frames | mask_area_mean | status | errors |
| --- | --- | --- | --- | --- | --- | --- |
| REAL_ENV249_00103_004_04 | REAL | small | 17/17/17 | 0.000000 | `EFFECTERASE_SMOKE_INPUTS_BLOCKED` | mask_empty |
| REAL_ENV231_00010_003_03 | REAL | medium | 17/17/17 | 0.092405 | `EFFECTERASE_SMOKE_INPUTS_READY` |  |
| REAL_ENV166_00002_001_02 | REAL | large | 17/17/17 | 0.277863 | `EFFECTERASE_SMOKE_INPUTS_READY` |  |
| BLENDER_FOREST026_00020 | BLENDER | small | 17/17/17 | 0.015635 | `EFFECTERASE_SMOKE_INPUTS_READY` |  |
| BLENDER_BEDROOM009_00083 | BLENDER | medium | 17/17/17 | 0.025008 | `EFFECTERASE_SMOKE_INPUTS_READY` |  |
| BLENDER_FOREST010_00004 | BLENDER | large | 17/17/17 | 0.274602 | `EFFECTERASE_SMOKE_INPUTS_READY` |  |

## Decision

The preregistered six-row smoke is blocked because at least one locked row is not technically valid.
The failing row is `REAL_ENV249_00103_004_04`, whose materialized mask is empty across all 17 frames.
Per preregistration rules, this row was not replaced, the seed/mask/frame indices were not changed, and no inference smoke was launched.

## PAI Output Paths

- `REAL_ENV249_00103_004_04` condition: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/REAL_ENV249_00103_004_04/fg_bg.mp4`
- `REAL_ENV249_00103_004_04` winner: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/REAL_ENV249_00103_004_04/bg.mp4`
- `REAL_ENV249_00103_004_04` mask: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/REAL_ENV249_00103_004_04/mask.mp4`
- `REAL_ENV231_00010_003_03` condition: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/REAL_ENV231_00010_003_03/fg_bg.mp4`
- `REAL_ENV231_00010_003_03` winner: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/REAL_ENV231_00010_003_03/bg.mp4`
- `REAL_ENV231_00010_003_03` mask: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/REAL_ENV231_00010_003_03/mask.mp4`
- `REAL_ENV166_00002_001_02` condition: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/REAL_ENV166_00002_001_02/fg_bg.mp4`
- `REAL_ENV166_00002_001_02` winner: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/REAL_ENV166_00002_001_02/bg.mp4`
- `REAL_ENV166_00002_001_02` mask: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/REAL_ENV166_00002_001_02/mask.mp4`
- `BLENDER_FOREST026_00020` condition: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/BLENDER_FOREST026_00020/fg_bg.mp4`
- `BLENDER_FOREST026_00020` winner: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/BLENDER_FOREST026_00020/bg.mp4`
- `BLENDER_FOREST026_00020` mask: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/BLENDER_FOREST026_00020/mask.mp4`
- `BLENDER_BEDROOM009_00083` condition: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/BLENDER_BEDROOM009_00083/fg_bg.mp4`
- `BLENDER_BEDROOM009_00083` winner: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/BLENDER_BEDROOM009_00083/bg.mp4`
- `BLENDER_BEDROOM009_00083` mask: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/BLENDER_BEDROOM009_00083/mask.mp4`
- `BLENDER_FOREST010_00004` condition: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/BLENDER_FOREST010_00004/fg_bg.mp4`
- `BLENDER_FOREST010_00004` winner: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/BLENDER_FOREST010_00004/bg.mp4`
- `BLENDER_FOREST010_00004` mask: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/BLENDER_FOREST010_00004/mask.mp4`
