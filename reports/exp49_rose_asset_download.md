# Exp49 ROSE Asset Download

Status: `ROSE_ASSETS_PARTIAL`

Generated: 2026-06-30T07:20:09,988600811+08:00
Host: dsw-753014-85f54df947-bkp7h
Branch: research/exp49-pai-rose-adapter-feasibility-20260629
Commit: 42fde51e60b2e1492d1e305673d587f357289677

## Method

PAI direct download was attempted. Direct `huggingface.co` calls timed out from PAI, so the PAI download used `https://hf-mirror.com`. H20 relay was not used.

## Paths

| Asset | Path |
| --- | --- |
| ROSE code | `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/ROSE/Kunbyte-AI_ROSE` |
| HF Space | `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/ROSE_HF_Space/Kunbyte_ROSE` |
| HF model | `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/rose/Kunbyte_ROSE` |
| Wan base model | `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/rose/Wan2.1-Fun-1.3B-InP` |
| ROSE dataset metadata | `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/rose_dataset/Kunbyte_ROSE_Dataset` |

## Asset Status

```text
asset	status	path	note
code	READY	/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/ROSE/Kunbyte-AI_ROSE	6be41c5420bf331c6d491277d5a6feaf9b3a779a
hf_space	READY	/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/ROSE_HF_Space/Kunbyte_ROSE	0ea1fc65605d8734bd85df2c12d8198687cc4229
hf_model	READY	/mnt/nas/hj/H20_Video_inpainting_DPO/weights/rose/Kunbyte_ROSE	hf mirror model download
wan_base	READY	/mnt/nas/hj/H20_Video_inpainting_DPO/weights/rose/Wan2.1-Fun-1.3B-InP	hf mirror base download
dataset_filelist	PARTIAL	/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp49_pai_rose_adapter_feasibility/rose_dataset_filelist.txt	file list unavailable or timed out
dataset_metadata	READY	/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/rose_dataset/Kunbyte_ROSE_Dataset	README/.gitattributes staged only
```

## Inventory And Checksums

- Inventory: `reports/exp49_rose_asset_inventory.txt`
- SHA256: `reports/exp49_rose_asset_sha256.txt`
- Summary JSON: `reports/exp49_rose_asset_download_summary.json`

Files over 128 MiB are listed as `SKIP_LARGE` in the checksum report to avoid blocking the milestone on multi-hour NAS hashing.

## Disk Snapshot

```text
Filesystem                                                                           Size  Used Avail Use% Mounted on
172.28.48.25:/                                                                        10P   10T   10P   1% /mnt/nas
rund:1FDr4OBr:cpfs-01000vwrt8a6usy68r6wu-000001.cn-shanghai.cpfs.aliyuncs.com:/pku/   70T   68T  2.3T  97% /mnt/workspace
172.28.48.25:/                                                                        10P   10T   10P   1% /mnt/nas
```

## Notes

No raw assets were added to Git. No GPU work, inference, training, or optimizer step was run in this milestone.
