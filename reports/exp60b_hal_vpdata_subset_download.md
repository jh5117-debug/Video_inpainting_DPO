# Exp60B HAL VPData Subset Download Fallback

Status: `EXP60B_HAL_VPDATA_SUBSET_BLOCKED`

## Route

- HAL host: local `/home/hj`
- Staging root considered: `/home/hj/vpdata_exp60b_hal_staging`
- Action taken: missing-URL network probe only
- GPU used: no
- Training / inference / loser generation: no

## Result

H20 hf-mirror plus H20 clash proxy downloaded 1,089/1,100 locked VPData rows. HAL fallback was required for the remaining 11 Pexels raw-video URLs. Before duplicating the already downloaded 1,089 files on HAL, the 11 missing URLs were range-probed from HAL with browser-like headers and Pexels referer.

- Missing URLs tested: 11
- HAL reachable URLs: 0
- HAL blocked URLs: 11
- Failure class: HTTP 403 from `videos.pexels.com`
- Full VPData downloaded: no
- Additional HAL raw videos downloaded: 0

Because every missing locked URL remained blocked from HAL, the exact train1000/test100 subset cannot be completed by HAL fallback in this continuation. PAI transfer is therefore not promoted to ready.

## Artifacts

- `reports/exp60b_hal_vpdata_subset_download.csv`
- `reports/exp60b_hal_vpdata_subset_summary.json`
- `reports/exp60b_hal_vpdata_subset_failed_urls.txt`
- `reports/exp60b_hal_vpdata_subset_sha256.txt`
- `reports/exp60b_hal_missing_url_probe.csv`

No token was printed or written. No data row was replaced.
