# Exp30 Smoke16 V2 Manifest Repair

Status: `EXP30_SMOKE16_V2_MANIFEST_REPAIRED_PRE_INFERENCE`

- Invalid sample IDs: `['BLENDER_CARTOON006_00001', 'REAL_ENV044_00004_001_01', 'REAL_ENV046_00001_001_01']`
- Replacement sample IDs: `['BLENDER_FOREST019_00001', 'REAL_ENV046_00004_001_01', 'REAL_ENV046_00005_001_01']`
- Rows: 16
- Source type counts: `{'BLENDER': 8, 'REAL': 8}`
- Final manifest: `exp30_vor_or_multimodel_minimax/manifests/vor_or_smoke16_v2_sources_final.jsonl`
- Final manifest SHA256: `7e8cfd1b672b17b131476c9dd82804841d22d7450adf26301cf9ae8ff83f7f76`

The repair happened before any model candidate output review. It is a technical decode/non-empty-mask replacement, not result-based source selection.
