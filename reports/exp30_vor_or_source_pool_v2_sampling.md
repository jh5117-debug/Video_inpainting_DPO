# Exp30 VOR-OR Source Pool V2 Sampling

Status: `VOR_OR_SOURCE_POOL_V2_READY`

## Inputs

- Metadata index: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp25_vor/exp25_vor_or_preference_data/manifests/vor_train_metadata_index.jsonl`
- Metadata SHA256: `33d57a3ea23c5799b583d476a311089f95cbce1b0d11280822a63b8c9edcddc4`
- Metadata rows after known-bad quarantine: 57750
- Scene groups after known-bad quarantine: 1449

## Exclusion

- Strict excluded scene groups: 209
- Strict excluded sample ids: 288
- Strict candidates after exclusion: 1240
- Relaxed diagnostic exclusion used: False

## Selected Pools

- Primary rows: 128
- Reserve rows: 128
- Reserve2 rows: 128
- Primary source type counts: `{'BLENDER': 64, 'REAL': 64}`
- Reserve source type counts: `{'BLENDER': 20, 'REAL': 108}`
- Reserve2 source type counts: `{'REAL': 128}`
- Mask buckets: metadata unavailable, recorded as `unknown` rather than inferred.
- Effect labels: metadata unavailable, recorded as `unknown` rather than inferred.

## Preview

No videos were extracted in this milestone. Preview status is
`metadata_only_visual_preview_pending`; full video evidence remains required
before any smoke pass, data-ready, or adapter-positive claim.

## Decision

Source-pool v2 meets the count gate: primary128 exists and reserve contains at least 64 rows. It unlocks multi-model OR smoke16, but does not itself constitute video-review or data-ready evidence.
