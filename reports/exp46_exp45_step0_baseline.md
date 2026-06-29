# Exp46 Exp45 Pseudo-Success Step0 Baseline

Status: EXP46_STEP0_BASELINE_READY

Step0 baseline evaluation completed on the H20 rewritten Exp45 pseudo-success search/shadow manifests.

- Raw MiniMax output is the primary baseline evidence.
- No training was run.
- No optimizer step was run.
- No GT-only SFT, DPO, VOR-Eval, or hard comp was used.
- Visual review contact sheets were opened for search24 and shadow24.

| split | rows | full PSNR | mask PSNR | boundary PSNR | outside PSNR | SSIM | LPIPS | Ewarp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| search | 24 | 36.518870 | 35.506194 | 28.336382 | 36.850255 | 0.990855 | N/A | 0.137264 |
| shadow | 24 | 35.636157 | 26.977549 | 25.474265 | 36.739831 | 0.978682 | N/A | 0.066924 |

Visual review:

- Search24: decoded and reviewed via temporal-strip and midframe contact sheets; no obvious black frames, global fogging, or outside destruction.
- Shadow24: decoded and reviewed via temporal-strip and midframe contact sheets; no obvious black frames, global fogging, or outside destruction.
- This is a baseline reference review only. It does not promote MiniMax or claim improvement.

Output files:

- Metrics: reports/exp46_exp45_step0_baseline_metrics.csv
- Visual review CSV: reports/exp46_exp45_step0_visual_review.csv
- Summary JSON: reports/exp46_exp45_step0_summary.json
- Review sheets: reports/exp46_step0_visual_montages/
