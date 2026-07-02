# Exp60C VPData Subset Transfer To PAI/NAS

Status: `EXP60C_TRANSFER_BLOCKED`

## Source Readiness

- H20 final subset status: `EXP60C_H20_VPDATA_SUBSET_READY`
- Train videos: `1000`
- Test videos: `100`
- Total videos: `1100`
- H20 sha256 rows: `1100`
- H20 OpenCV decode after targeted repair: `1100 / 1100`
- Full VPData downloaded: no

## Transfer Attempt

Target requested by protocol:

`/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/vpdata_exp60b/raw_subset`

PAI user:

`hj@47.103.26.60`

The transfer did not start because the PAI user cannot create the target
directory:

`Permission denied`

Permission audit showed `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external`
is owned by root and is not writable by `hj`. The `/mnt/workspace/hj/nas_hj`
alias points to the same unwritable NAS tree.

## Decision

No PAI ready manifest was generated. No `EXP60C_PAI_VPDATA_SUBSET_READY` status
was written. The exact blocker is NAS target write permission, not H20 data
readiness.

No masks, losers, inference, DPO, training, GPU job, or full VPData download
ran.
