# Exp60C Failed URL Replacement Readback

Status: `EXP60C_FAILED_ROWS_AUDITED`

## Git / Source State

- Branch: `research/exp60b-videopainter-vpdata-d3mask-pai-20260702`
- Required start HEAD: `efed1f8adad1de355c4f27a5843cfc058765af75`
- Worktree readback: clean before Exp60C reports were generated

## H20 Staging State

- Existing successful raw videos on H20: `1089`
- H20 staging size: about `14G`
- Existing successful videos will be kept and must not be redownloaded.

## Failed Rows

- Original failed URLs: `11`
- Train failed rows: `9`
- Test failed rows: `2`
- Reason: `HTTP 403 Forbidden` from `videos.pexels.com`
- H20 hf-mirror route: failed for the same 11 URLs
- H20 clash proxy route: failed for the same 11 URLs
- HAL fallback probe: failed for the same 11 URLs

## SHA256

- H20 hf-mirror SHA256 rows: `1089`
- H20 proxy SHA256 rows: `1089`
- Complete 1,100-row SHA256 set: no

## Safety

No masks, losers, DPO data, inference, training, GPU job, or full VPData download has run.

## Artifacts

- `reports/exp60c_failed_rows_audit.csv`
- `reports/exp60c_start_state_summary.json`
