# Exp30 Gate64 V3 Manifest Repair

Status: `EXP30_GATE64_V3_MANIFEST_REPAIRED_PRE_INFERENCE`

- Original rows: 64
- Pre-inference materialization failures: 9
- Replacement rows: 9
- Final rows: 64
- Final source type counts: `{'BLENDER': 32, 'REAL': 32}`
- Final manifest: `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_sources_v3_final.jsonl`
- Final manifest SHA256: `c2da063118934f0b03d13d88015cfc1cc57e881aca257307ca42de20cc944eb0`

The repair happened before any Gate64 model output, visual selection,
adapter gate, or training.  Failed rows are preserved in the report and
not silently reused.
