# PRD 22: This Week Best Result And Visual Evidence

Date: 2026-06-15

## Current Best

The current best method is:

```text
Exp11 boundary outer b0.75 S2
stage combination = DPO-S1 + DPO-S2
```

This is not the old Exp11-proxy. It is the region / boundary ablation line:

- boundary mode: `outer`
- mask weight: `1.0`
- boundary weight: `0.75`
- outside weight: `0.05`
- gap normalization: `log_ratio`
- loss region mode: `region`

## Fixed Metric Protocol

All claims use the fixed DAVIS50 protocol:

```text
raw6
D+G off
no PCM
no mask dilation
no Gaussian blur
hard comp
frame-wise in-memory metric
metric backend = inference/metrics.py via tools/run_davis50_framewise_protocol_eval.py
```

## Metric Summary

| Method | Stage | PSNR | SSIM | LPIPS | VFID | TC | Mask PSNR |
|---|---|---:|---:|---:|---:|---:|---:|
| SFT-48000 baseline | SFT base | 32.731391 | 0.970533 | 0.016660 | 0.201792 | 0.971200 | 23.884924 |
| Exp11 boundary outer b0.75 | DPO-S1 + SFT-S2 | 32.901188 | 0.971859 | 0.015104 | 0.188015 | 0.971287 | 24.054721 |
| Exp11 boundary outer b0.75 | DPO-S1 + DPO-S2 | 33.013954 | 0.972295 | 0.015363 | 0.175423 | 0.971122 | 24.167487 |

Exp12 adaptive normalization did not beat this result under the same protocol.

## Verified Visual Evidence

HAL archive:

```text
/home/hj/dpo-2-1-exp/this_week_exp11_exp12/visual_evidence/exp11_outer_b075_s2_selected_visuals
```

PAI source:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/20260615_exp11_outer_b075_s2_selected_visuals
```

Verified contents:

- side-by-side MP4 files
- frame-by-frame JPGs
- contact sheets
- SFT-48000 / Exp11 per-video metrics
- visual evidence manifest
- README

Best cases:

- strongest: `boat`
- usable positive: `rhino`, `dog-agility`, `lucia`, `blackswan`
- caution / failure: `dance-jump`, `soccerball`

`boat` is the clearest paper/PPT example: the SFT-48000 baseline shows a
visible white fog / patch around the wake and hull, while Exp11 outer b0.75 S2
keeps cleaner water texture and boundary continuity.

## DPO Diagnostics

Diagnostics:

```text
/home/hj/dpo-2-1-exp/this_week_exp11_exp12/dpo_diag/exp11_boundary_outer_b075_s1_dpo_diagnostics.csv
/home/hj/dpo-2-1-exp/this_week_exp11_exp12/dpo_diag/exp11_boundary_outer_b075_s2_dpo_diagnostics.csv
```

Summary:

- Stage1 label: `LOSER_DOMINANT`
- Stage2 label: `LOSER_DOMINANT`
- No old-style raw-DPO winner-gap explosion.
- Report the result as metric + visual improvement with residual loser-dominant
  diagnostic risk.

## Reports

```text
reports/exp11_outer_b075_s2_visual_evidence_report.md
reports/exp11_outer_b075_s2_visual_evidence_verification.md
reports/exp11_outer_b075_s2_dpo_diag_summary.md
```

## Final Paper/PPT Case Pool

Final visual package:

```text
/home/hj/dpo-2-1-exp/final_20_visual_cases_for_paper
```

The package contains:

- 20 four-column side-by-side MP4 files
- 20 frame contact sheets
- selected per-frame JPG panels
- `final_20_visual_cases_for_paper.csv`
- `README.md`

Composition:

- DAVIS50 positives: `boat`, `rhino`, `dog-agility`, `lucia`, `blackswan`
- YouTubeVOS100 positives: top 15 metric-gain candidates after contact-sheet
  sanity review.

Report:

```text
reports/final_20_visual_cases_for_paper.md
reports/final_20_visual_cases_for_paper.csv
```
