# Exp29 EffectErase Smoke V2 Input Audit

Status: `EFFECTERASE_SMOKE_V2_PREREGISTERED`

Old manifest SHA256: `54fd62a97fa69f2f17590488136d426cee77de0ed02548c46a83d8818be2b137`
New manifest SHA256: `b16a0007a22f190bb7894a673092063efb5dd2eda26dbd53737cdc987d9d4f36`
Rejected old row: `REAL_ENV249_00103_004_04` because its smoke materialized mask is empty across all 17 frames.
Replacement row: `REAL_ENV248_00118_005_03`

## Accepted Rows

| sample_id | source_type | mask_bucket | non-empty mask frames | median mask area | masked absdiff mean |
| --- | --- | --- | ---: | ---: | ---: |
| REAL_ENV231_00010_003_03 | REAL | medium | 17/17 | 0.098274 | 85.021 |
| REAL_ENV166_00002_001_02 | REAL | large | 17/17 | 0.285427 | 35.620 |
| BLENDER_FOREST026_00020 | BLENDER | small | 17/17 | 0.015556 | 36.887 |
| BLENDER_BEDROOM009_00083 | BLENDER | medium | 17/17 | 0.025024 | 65.780 |
| BLENDER_FOREST010_00004 | BLENDER | large | 17/17 | 0.275192 | 95.232 |
| REAL_ENV248_00118_005_03 | REAL | small | 17/17 | 0.034763 | 125.437 |

All accepted rows are tagged `diagnostic_only_vor_confounded`, `eligible_for_training=false`, and `vor_eval=false`.

Preview sheets were generated under `reports/exp29_effecterase_smoke_v2_previews/` and inspected with `view_image` for all six rows. Each row shows a non-empty task mask, a decodable condition/winner pair, and visible condition-vs-winner difference in the target/effect region.
