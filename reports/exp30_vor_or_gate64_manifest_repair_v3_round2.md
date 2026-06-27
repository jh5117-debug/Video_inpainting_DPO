# Exp30 Gate64 V3 Manifest Repair

Status: `EXP30_GATE64_V3_MANIFEST_REPAIRED_PRE_INFERENCE`

- Original rows: 64
- Pre-inference materialization failures: 1
- Replacement rows: 1
- Final rows: 64
- Final source type counts: `{'BLENDER': 32, 'REAL': 32}`
- Final manifest: `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_sources_v3_final_round2.jsonl`
- Final manifest SHA256: `e702fd49d205e2599648f55e768bda4b474a23061fdc9a967d0656c530e30a25`

The repair happened before any Gate64 model output, visual selection,
adapter gate, or training.  Failed rows are preserved in the report and
not silently reused.
