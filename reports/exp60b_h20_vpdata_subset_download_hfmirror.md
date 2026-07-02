# Exp60B H20 VPData Subset Download With HF Mirror

Status: `EXP60B_H20_PEXELS_RAW_PROXY_REQUIRED`

## Route

- H20 route: HAL -> PAI -> H20
- H20 hostname: `instance-afs92r3e`
- Worktree: `/home/nvme01/H20_Video_inpainting_DPO_exp60b_vp_vpdata_transfer`
- HF endpoint: `https://hf-mirror.com`
- Workers: 8
- GPU used: no
- Training / inference / loser generation: no

## Result

- Selected rows: 1,100
- Train rows: 1,000
- Test rows: 100
- Downloaded raw videos: 1,089
- Failed raw URLs: 11
- Full VPData downloaded: no
- H20 staging size after attempt: about 14G
- SHA256 rows written: 1,089

The Hugging Face metadata path was successfully unblocked through
`hf-mirror.com`. The remaining failures are Pexels raw-video URL failures, so
the next required milestone is H20 clash proxy fallback for the failed URLs /
resume set.

## Stale Process Note

An earlier serial Exp60B download process survived a local interrupt and was
terminated by PGID on H20. The active parallel process was left running and
completed the hf-mirror attempt.

## Artifacts

- `reports/exp60b_h20_vpdata_subset_download_hfmirror.csv`
- `reports/exp60b_h20_vpdata_subset_hfmirror_summary.json`
- `reports/exp60b_h20_vpdata_subset_hfmirror_failed_urls.txt`
- `reports/exp60b_h20_vpdata_subset_hfmirror_sha256.txt`

No token was printed or written.

