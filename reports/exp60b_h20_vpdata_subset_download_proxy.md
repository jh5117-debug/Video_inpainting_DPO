# Exp60B H20 VPData Subset Download With Clash Proxy

Status: `EXP60B_H20_VPDATA_SUBSET_BLOCKED_PROXY`

## Route

- H20 route: HAL -> PAI -> H20
- H20 hostname: `instance-afs92r3e`
- Worktree: `/home/nvme01/H20_Video_inpainting_DPO_exp60b_vp_vpdata_transfer`
- Proxy route: H20 clash enabled through `/home/nvme01/clash-for-linux/clash.sh`
- HF endpoint used under proxy: `https://huggingface.co`
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
- SHA256 rows written: 1,089

The clash proxy fallback did not recover the remaining Pexels raw-video URLs.
All 11 remaining failures are still HTTP 403 from `videos.pexels.com`.
Therefore H20 mirror plus H20 proxy did not produce the exact locked
train1000/test100 subset, and HAL fallback is required before PAI transfer can
be marked ready.

## Notes

- No token was printed or written.
- No extra VPData rows were added.
- No full VPData clone/download was attempted.
- The downloader reports `FFPROBE_ERROR` for downloaded rows because `ffprobe`
  is absent on H20; this is a decode-tool availability issue, not a download
  completion signal. PAI/NAS verification must still run decode spot checks.

## Artifacts

- `reports/exp60b_h20_vpdata_subset_download_proxy.csv`
- `reports/exp60b_h20_vpdata_subset_proxy_summary.json`
- `reports/exp60b_h20_vpdata_subset_proxy_failed_urls.txt`
- `reports/exp60b_h20_vpdata_subset_proxy_sha256.txt`
