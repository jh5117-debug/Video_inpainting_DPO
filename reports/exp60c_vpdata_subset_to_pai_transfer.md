# Exp60C VPData Subset Transfer To PAI/NAS

Status: `EXP60C_PAI_VPDATA_SUBSET_READY`

## Permission Recovery

PAI/NAS target permission was already fixed before this milestone. Codex did not run any root permission script, chmod, or chown. The target directories were verified as writable by `hj` in `reports/exp60c_pai_target_permission_recovery.*`.

## Transfer

- H20 source: `/home/nvme01/H20_Video_inpainting_DPO/data/external/vpdata_exp60b_h20_staging/raw_subset`
- PAI/NAS target: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/vpdata_exp60b/raw_subset`
- Transfer command class: `rsync -aH --partial --append-verify` from PAI pulling H20 over SSH
- Raw MP4 count on PAI: `1100`
- Raw total size on PAI: `14460346432` bytes (~13.47 GiB / 14G by `du -sh`)
- Full VPData downloaded: no
- Replacement / URL download rerun: no

## H20 Source Readiness

- H20 final subset status: `EXP60C_H20_VPDATA_SUBSET_READY`
- Train videos: `1000`
- Test videos: `100`
- Total videos: `1100`
- H20 sha256 rows: `1100`
- H20 OpenCV decode after targeted repair: `1100 / 1100`

## PAI Verification

- PAI sha256 status: `SHA256_MATCH`
- PAI sha256 rows: `1100`
- PAI OpenCV decode: `1100 / 1100`
- PAI manifest train count: `1000`
- PAI manifest test count: `100`
- PAI train/test source_video_id overlap: `0`
- PAI train/test URL overlap: `0`
- Duplicate source_video_id: `0`
- Duplicate source_url: `0`
- Duplicate planned_video_path: `0`
- H20/HAL local path violations in PAI manifests: `0`

PAI manifests generated:

- `manifests/exp60c_vpdata_train1000_sources_pai.jsonl`
- `manifests/exp60c_vpdata_test100_sources_pai.jsonl`

PAI/NAS copies:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/vpdata_exp60b/manifests/exp60c_vpdata_train1000_sources_pai.jsonl`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/vpdata_exp60b/manifests/exp60c_vpdata_test100_sources_pai.jsonl`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/vpdata_exp60b/sha256/exp60c_pai_final_subset_sha256.txt`

## Metadata / Captions / Native Masks

- Captions are preserved in both PAI manifests for all 1,100 rows.
- VPData `native_mask_path` references are preserved in both PAI manifests for all 1,100 rows.
- No native mask files were materialized in this milestone; mask generation remains intentionally not run.
- No non-MP4 files are present under the PAI raw video subset.

## Safety

No masks, losers, inference, DPO, training, GPU job, full VPData download, or token exposure occurred in this milestone.

Generated: `2026-07-02T16:48:03.821453+02:00`
