# Exp60C VPData Download Final Decision

Status: `VIDEOPAINTER_VPDATA_DOWNLOAD_BLOCKED_STORAGE`

## Required Answers

1. The original 11 failed HTTP-403 URLs were replaced by deterministic
   same-split Pexels-only replacement rows.
2. The final H20 subset has train1000 / test100.
3. Full VPData was avoided.
4. Files were not transferred to the requested PAI/NAS target because `hj`
   lacks write permission under `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external`.
5. H20 SHA256 is complete for 1,100 files. PAI SHA256 match is not available
   because transfer was blocked before data copy.
6. PAI manifests are not ready.
7. Exp60B/60C is not yet unblocked for PAI D3 mask generation.
8. Exact blocker: PAI/NAS target storage permission, not source URL availability.
9. No VPData validation claim is made.

## H20 Result

- Replacement rows needed: `11`
- Replacement rows downloaded: `11`
- Backup attempts beyond rank-1: `0`
- Initial H20 decode: `1094 / 1100`
- Targeted corrupt-file repair: `6 / 6`
- Final H20 decode: `1100 / 1100`
- Final H20 sha256 rows: `1100`

## Boundary

No masks, losers, inference, DPO, training, GPU use, full VPData download, or
token logging occurred.
