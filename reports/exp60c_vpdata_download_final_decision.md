# Exp60C VPData Download Final Decision

Status: `VIDEOPAINTER_VPDATA_SUBSET_READY_ON_PAI`

## Required Answers

1. The original 11 failed HTTP-403 URLs were replaced by deterministic same-split Pexels-only replacement rows.
2. The final subset still has train1000 / test100.
3. Full VPData was avoided.
4. Files were transferred to the requested PAI/NAS target.
5. PAI SHA256 matched for 1,100 / 1,100 files.
6. PAI manifests are ready:
   - `manifests/exp60c_vpdata_train1000_sources_pai.jsonl`
   - `manifests/exp60c_vpdata_test100_sources_pai.jsonl`
7. Exp60B/60C is now unblocked for a separate PAI D3 mask generation milestone.
8. Exact blocker: none for raw VPData subset transfer. Mask generation has not started and remains separately gated.
9. No VPData validation claim is made.

## H20 Result

- Replacement rows needed: `11`
- Replacement rows downloaded: `11`
- Backup attempts beyond rank-1: `0`
- Initial H20 decode: `1094 / 1100`
- Targeted corrupt-file repair: `6 / 6`
- Final H20 decode: `1100 / 1100`
- Final H20 sha256 rows: `1100`

## PAI Result

- PAI target: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/vpdata_exp60b/raw_subset`
- Raw MP4s: `1100`
- Size: `14460346432` bytes (`14G` by `du -sh`)
- Train rows: `1000`
- Test rows: `100`
- SHA256: `SHA256_MATCH` (`1100 / 1100`)
- Decode: `1100 / 1100` OpenCV pass
- Train/test source overlap: `0`
- Train/test URL overlap: `0`
- Duplicate source IDs / URLs / paths: `0 / 0 / 0`
- H20/HAL local path leakage in PAI manifests: `0`

## Boundary

No masks, losers, inference, DPO, training, GPU use, full VPData download, token logging, or VPData positive claim occurred.

Generated: `2026-07-02T16:51:56.244308+02:00`
