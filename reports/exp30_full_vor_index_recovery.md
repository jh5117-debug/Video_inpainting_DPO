# Exp30 Full VOR Index Recovery

Status: `FULL_VOR_VALID_TRIPLET_INDEX_READY`

## Source Of Truth

- Metadata index: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp25_vor/exp25_vor_or_preference_data/manifests/vor_train_metadata_index.jsonl`
- Metadata SHA256: `33d57a3ea23c5799b583d476a311089f95cbce1b0d11280822a63b8c9edcddc4`
- Metadata rows: 57751
- Valid triplets after quarantining known bad rows: 57750
- Known bad quarantined rows: 1
- Scene groups: 1449
- Source type counts: `{'BLENDER': 21495, 'REAL': 36256}`

## Pairing Rule

- condition: `VOR-Train/FG_BG/<sample_id>.mp4`
- winner: `VOR-Train/BG/<sample_id>.mp4`
- mask: `MASK/<sample_id>.mp4`

## Member Index

- Member index: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp25_vor/reports/vor_train_mask_member_index.csv`
- Member SHA256: `3b3415c989e72b4df821f85903d01a754fb2c07412e40907749bf9844626d1f8`
- Member audit: `{'available': True, 'row_count': 179189, 'file_count': 179183, 'unsafe_count': 0, 'fields': ['group', 'member_index', 'member_path', 'sample_id', 'type', 'size', 'mtime', 'unsafe_reason'], 'type_counts': {'dir': 6, 'file': 179183}, 'prefix_counts': {'VOR-Train/BG': 61208, 'VOR-Train/FG_BG': 57751, 'MASK': 60224}}`

## Safety

- VOR tar archives scanned: no
- Videos extracted: no
- VOR-Eval used: no
- Exp25 worktree modified: no

## Decision

The full VOR metadata index is a usable source-of-truth for Exp30 source-pool v2 sampling. The previous 192-triplet result was an exact-extraction-cache subset, not the full VOR-Train inventory.
