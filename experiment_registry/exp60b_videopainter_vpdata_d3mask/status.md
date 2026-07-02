# Exp60B Status

Status:

- `EXP60B_READBACK_DONE`
- `EXP60B_VPDATA_AVAILABLE`
- `EXP60B_PAI_GPU_READY`
- `EXP60B_H20_READY_VIA_PAI_RELAY`
- `EXP60B_VPDATA_SUBSET_PLAN_READY`
- `EXP60B_H20_VPDATA_SUBSET_BLOCKED_NETWORK`
- `EXP60B_H20_HF_MIRROR_READY`
- `EXP60B_H20_PEXELS_RAW_PROXY_REQUIRED`
- `EXP60B_H20_VPDATA_SUBSET_BLOCKED_PROXY`
- `EXP60B_HAL_VPDATA_SUBSET_BLOCKED`
- `EXP60C_FAILED_ROWS_AUDITED`
- `EXP60C_REPLACEMENT_PLAN_READY`
- `EXP60C_REPLACEMENT_DOWNLOAD_READY`
- `EXP60C_H20_VPDATA_SUBSET_READY`
- `EXP60C_TRANSFER_BLOCKED`
- `EXP60C_PAI_TARGET_PERMISSION_RECOVERED`

Current continuation: H20 hf-mirror download completed partially: 1,089/1,100
raw videos downloaded, 11 Pexels raw URLs failed. H20 clash proxy fallback was
then run with resume and remained at 1,089/1,100, with the same 11 Pexels raw
URLs blocked by HTTP 403. HAL fallback probed those 11 missing locked URLs and
also received HTTP 403 for all 11. The exact train1000/test100 subset is still
blocked; PAI ready transfer is not allowed yet. Full VPData clone/download
remains forbidden.

Exp60C continuation replaced the 11 blocked source URLs with deterministic
same-split Pexels-only rows. H20 now has a complete 1,100-video subset with
1,100 sha256 rows and 1,100/1,100 OpenCV decode pass after targeted repair of 6
incomplete historical MP4 files. PAI/NAS transfer is blocked because the target
NAS data directory is root-owned and not writable by user `hj`.

Root-side permission recovery has now been applied. Codex verified the PAI/NAS
target dirs exist, are owned by `hj:hj`, have mode `0770`, and pass read/write/
execute/write_probe checks. Next milestone is transfer and PAI verification.
