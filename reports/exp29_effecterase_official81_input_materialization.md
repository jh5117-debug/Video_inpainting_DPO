# Exp29 EffectErase Official 81F Input Materialization

Status: `EFFECTERASE_OFFICIAL81_INPUTS_READY`

- Manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_official81_source_audit_20260627/manifests/effecterase_smoke_official81_preregistered.jsonl`
- Manifest SHA256: `706cb09286fd8528d7efbbb91eb89673a9ec7ce61b0047e6b3b2e8ea4c9b1fb3`
- Rows: 8
- Ready rows: 8
- Blocked rows: []
- Resolution: 832x480
- Frames per stream: 81
- VOR-Eval use: False
- Training eligibility: False

| sample_id | condition/winner/mask frames | resolution | non-empty mask frames | mask area median | status | errors |
| --- | --- | --- | ---: | ---: | --- | --- |
| REAL_ENV005_00003_003_05 | 81/81/81 | 832x480 | 81 | 0.021762 | `EFFECTERASE_OFFICIAL81_INPUTS_READY` |  |
| BLENDER_CON001_00218 | 81/81/81 | 832x480 | 81 | 0.025165 | `EFFECTERASE_OFFICIAL81_INPUTS_READY` |  |
| REAL_ENV024_00002_008_01 | 81/81/81 | 832x480 | 81 | 0.060945 | `EFFECTERASE_OFFICIAL81_INPUTS_READY` |  |
| BLENDER_BEACH036_00001 | 81/81/81 | 832x480 | 81 | 0.042450 | `EFFECTERASE_OFFICIAL81_INPUTS_READY` |  |
| REAL_ENV026_00001_002_02 | 81/81/81 | 832x480 | 81 | 0.130707 | `EFFECTERASE_OFFICIAL81_INPUTS_READY` |  |
| BLENDER_BEACH030_00003 | 81/81/81 | 832x480 | 81 | 0.121777 | `EFFECTERASE_OFFICIAL81_INPUTS_READY` |  |
| REAL_ENV097_00001_002_02 | 81/81/81 | 832x480 | 81 | 0.003062 | `EFFECTERASE_OFFICIAL81_INPUTS_READY` |  |
| REAL_ENV102_00001_002_02 | 81/81/81 | 832x480 | 81 | 0.080076 | `EFFECTERASE_OFFICIAL81_INPUTS_READY` |  |

Materialized preview sheets were generated for input sanity review.
No EffectErase inference was launched by this materialization milestone.

## Codex Materialized Preview Review

Codex opened all 8 materialized preview sheets on 2026-06-27. The 832x480
condition, winner, and mask-overlay strips remain aligned and decode as 81-frame
inputs. No empty-mask, frame-order, resize, or encoding failure was observed in
the sampled strips. Detailed row decisions are recorded in
`reports/exp29_effecterase_official81_materialized_preview_review.csv`.
