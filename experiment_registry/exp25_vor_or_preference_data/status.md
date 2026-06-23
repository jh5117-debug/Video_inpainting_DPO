# Exp25 VOR OR Preference Data

- repo: FudanCVL/EffectErase
- HF authenticated user: JiaHuang01
- dataset revision: `fa09dc61128ca0418a4a13364d97a08018ea9cc7`
- PAI outbound HF network: unavailable; use HAL-only download then rsync to PAI.
- core scope: README, VOR-Eval parts, VOR-Train-MASK parts, VOR-Train parts.
- excluded this round: VOR-Wild.
- required files: 37
- required total bytes: 363730944386
- largest part bytes: 10737418240
- HAL staging: `/home/hj/exp25_effecterase_staging`
- HAL free bytes at selection: 571536965632
- PAI destination: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/downloads/fa09dc61128ca0418a4a13364d97a08018ea9cc7`
- status: CORE_DOWNLOAD_COMPLETE
- completed time: 2026-06-23T00:08:45+0200
- completed files: 37 / 37
- completed bytes: 363730944386 / 363730944386
- PAI final files: 37
- PAI partial files: 0
- PAI bad files: 0
- HAL staging final size: 1.0K

## Safety

This track is download-only. It does not enter Exp23 worktrees, use GPUs, run inference, generate losers, or start DPO training. Tokens remain only under `/home/hj/.cache/huggingface_effecterase_auth` and are not copied to PAI or committed.

## Completion

Core compressed files are present under the fixed dataset revision directory on PAI:

`/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/downloads/fa09dc61128ca0418a4a13364d97a08018ea9cc7`

The PAI completion marker exists at:

`/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/runtime/CORE_DOWNLOAD_COMPLETE`

The transfer manifest contains 37 VERIFIED rows and zero HAL/PAI SHA256 mismatches. Final independent PAI inventory verification completed with `ok=true`: 37 OK files, 0 partial files, 0 bad files, 363730944386 final bytes, and contiguous VOR-Train archive parts 000-031.

## Current Work

Status: `SELECTIVE_OR_DATA_TOOLING_STARTED`

Added initial Exp25-only scripts for lightweight archive inspection, resumable
member indexing, safe selective extraction, extracted-subset validation, and
canonical VOR OR manifest semantics. No VOR archive was decompressed in full,
no loser generation was started, and no GPU training was launched in this step.

Archive audit: `reports/vor_archive_integrity.md` passed on PAI with matching
part counts and byte counts for VOR-Eval, VOR-Train-MASK, and VOR-Train. Stream
probe opened all three split gzip/tar streams and found zero unsafe paths in
the probed members.

VOR-Eval extraction: completed on PAI under
`/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/extracted/vor_eval_full`.
`BG`, `FG_BG`, and `MASK` each contain 43 aligned mp4 files and are reserved for
held-out final evaluation only.
