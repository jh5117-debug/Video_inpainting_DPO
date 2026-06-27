# Exp30 Gate64 V3 Manifest Repair

Status: `EXP30_GATE64_V3_MANIFEST_REPAIRED_PRE_INFERENCE`

- Original rows: 64
- Pre-inference materialization failures: 1
- Replacement rows: 1
- Final rows: 64
- Final source type counts: `{'BLENDER': 32, 'REAL': 32}`
- Final manifest: `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_sources_v3_final_round3.jsonl`
- Final manifest SHA256: `1fa520151084ba7613fe8afe51e82bd66ba15fa145d3fa9e3c50b173e173fa5c`

The repair happened before any Gate64 model output, visual selection,
adapter gate, or training.  Failed rows are preserved in the report and
not silently reused.
