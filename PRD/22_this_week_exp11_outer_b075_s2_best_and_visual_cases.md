# PRD 22: This Week Exp11 Outer B0.75 S2 Best Result And Visual Cases

Date: 2026-06-15

## Current Best

The current best method is:

```text
Exp11 boundary outer b0.75 S2
stage combination = DPO-S1 + DPO-S2
```

This is the region / boundary ablation line, not the old Exp11-proxy:

- boundary mode: `outer`
- mask weight: `1.0`
- boundary weight: `0.75`
- outside weight: `0.05`
- gap normalization: `log_ratio`
- loss region mode: `region`

It is currently the best result under the fixed DAVIS50 raw6 hard-comp protocol.

## Fixed Metric Protocol

All paper-facing claims use the fixed inpainting protocol:

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

This protocol is now the only accepted default for DAVIS / YouTubeVOS inpainting evaluation in this project.

## DAVIS50 Metric Summary

| Method | Stage | PSNR | SSIM | LPIPS | VFID | TC | Mask PSNR |
|---|---|---:|---:|---:|---:|---:|---:|
| SFT-48000 baseline | SFT base | 32.731391 | 0.970533 | 0.016660 | 0.201792 | 0.971200 | 23.884924 |
| Exp11 boundary outer b0.75 | DPO-S1 + SFT-S2 | 32.901188 | 0.971859 | 0.015104 | 0.188015 | 0.971287 | 24.054721 |
| Exp11 boundary outer b0.75 | DPO-S1 + DPO-S2 | 33.013954 | 0.972295 | 0.015363 | 0.175423 | 0.971122 | 24.167487 |

Exp12 adaptive normalization did not beat Exp11 outer b0.75 S2 under the same protocol, so Exp12 remains an ablation / negative comparison rather than the final method.

## DAVIS50 + YouTubeVOS100 Extended Summary

| Dataset | Method | Rows | PSNR | SSIM | LPIPS | VFID | TC | Mask PSNR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DAVIS50 | SFT-48000 | 50 | 32.7314 | 0.9705 | 0.0167 | 0.2018 | 0.9712 | 23.8849 |
| DAVIS50 | Exp11 outer b0.75 S2 | 50 | 33.0140 | 0.9723 | 0.0154 | 0.1754 | 0.9711 | 24.1675 |
| YouTubeVOS100 | SFT-48000 | 100 | 33.3968 | 0.9701 | 0.0176 | 0.2007 | 0.9819 | 24.4262 |
| YouTubeVOS100 | Exp11 outer b0.75 S2 | 100 | 33.7238 | 0.9711 | 0.0168 | 0.1925 | 0.9821 | 24.7532 |

Exp11 outer b0.75 S2 improves over SFT-48000 on both DAVIS50 and YouTubeVOS100. Ewarp is not present in the current consolidated summary; PSNR / SSIM / LPIPS / VFID / TC / mask PSNR are the available reported metrics.

Report paths:

```text
reports/exp11_outer_b075_s2_youtubevos100_davis50_eval.md
reports/exp11_outer_b075_s2_youtubevos100_davis50_eval.csv
```

## Selected Success / Failure Evidence

Archive:

```text
/home/hj/dpo-2-1-exp/this_week_exp11_exp12/visual_evidence/exp11_outer_b075_s2_selected_visuals
```

Strongest positive:

- `boat`: SFT-48000 has visible white fog / patch around the wake and hull; Exp11 outer b0.75 S2 keeps cleaner water texture and boundary continuity. This is the lead paper/PPT case.

Usable positive cases:

- `rhino`: animal boundary and foreground/background separation improve.
- `dog-agility`: useful motion and thin-structure case; PSNR improves, mask SSIM is mixed.
- `lucia`: subtle human-case improvement; best as supplementary evidence.
- `blackswan`: mild animal/water boundary improvement; mask SSIM is mixed.

Caution / failure cases:

- `dance-jump`: per-video PSNR / mask PSNR and mask SSIM are below baseline; do not use as a positive figure.
- `soccerball`: per-video metric is below baseline; useful only as a failure/caution case.

## Final 20 Paper/PPT-Ready Cases

Package:

```text
/home/hj/dpo-2-1-exp/final_20_visual_cases_for_paper
```

Contents:

- GT / winner
- mask overlay
- SFT-48000 DiffuEraser baseline
- Exp11 outer b0.75 S2
- side-by-side / four-column videos
- frame-by-frame contact sheets
- per-video metric rows

Summary reports:

```text
reports/final_20_visual_cases_for_paper_summary.md
reports/final_20_visual_cases_for_paper_summary.csv
reports/final_20_visual_cases_for_paper.md
reports/final_20_visual_cases_for_paper.csv
```

| # | Dataset | Video | Why show it | Base problem | Exp11 improvement | Remaining issue | Metric |
|---:|---|---|---|---|---|---|---|
| 1 | DAVIS50 | `boat` | strongest visual: cleaner wake/hull boundary | SFT-48000 在船身和尾浪附近有白雾/贴片感，水面纹理和 hull 边界不稳定。 | Exp11 outer b0.75 S2 明显压掉白雾，水面纹理和船体边界更连续。 | 仍有轻微纹理平滑，但它是当前最强正例，适合主图。 | dPSNR 1.4362, dMaskPSNR 1.4362 |
| 2 | DAVIS50 | `rhino` | foreground animal mask boundary improves | SFT-48000 在动物前景边界附近有涂抹和背景黏连。 | Exp11 对 mask 外边界施加更强约束后，前景/背景分离更干净。 | 毛发/皮肤细节仍不是完美恢复，适合正例或补充图。 | dPSNR 0.9862, dMaskPSNR 0.9862 |
| 3 | DAVIS50 | `dog-agility` | large metric gain and useful motion case | SFT-48000 在高速运动和细杆结构附近容易出现边界模糊。 | Exp11 提升整体和 mask PSNR，运动边界更稳。 | mask SSIM 曾有混合信号，适合做运动类正例但不要单独夸 SSIM。 | dPSNR 0.7001, dMaskPSNR 0.7001 |
| 4 | DAVIS50 | `lucia` | subtle positive human case | SFT-48000 的人物区域补全较稳，但边界和局部纹理还有轻微不自然。 | Exp11 有小幅但一致的 PSNR/mask PSNR 提升，人物边界更平顺。 | 视觉改善偏 subtle，适合补充材料。 | dPSNR 0.3976, dMaskPSNR 0.3976 |
| 5 | DAVIS50 | `blackswan` | mild positive water/animal boundary case | SFT-48000 在黑天鹅和水面交界处有轻微水纹/边界不连续。 | Exp11 对水面和动物边界更稳，整体 PSNR 提升。 | mask SSIM 混合，作为温和正例更合适。 | dPSNR 0.4102, dMaskPSNR 0.4102 |
| 6 | YouTubeVOS100 | `5b33c701ce` | top YouTubeVOS gain candidate: dPSNR=5.018, dSSIM=0.0016, dLPIPS=0.0022 | SFT-48000 在 person/animal outdoor 场景的 mask 区域容易出现局部模糊、贴片感或边界过渡不自然。 | Exp11 在该视频上 whole/mask PSNR 均提升 5.0176，说明外边界加权对补洞区域和上下文连续性有帮助。 | 该样本已通过 contact sheet sanity review，但仍建议在论文图中逐帧确认语义是否自然。 LPIPS 略差，作为定性图时需搭配 PSNR/SSIM 说明。 | dPSNR 5.0176, dMaskPSNR 5.0176, dLPIPS 0.0022 |
| 7 | YouTubeVOS100 | `8d55a5aebb` | top YouTubeVOS gain candidate: dPSNR=3.358, dSSIM=0.0035, dLPIPS=0.0003 | SFT-48000 在 person/object indoor 场景的 mask 区域容易出现局部模糊、贴片感或边界过渡不自然。 | Exp11 在该视频上 whole/mask PSNR 均提升 3.3585，说明外边界加权对补洞区域和上下文连续性有帮助。 | 该样本已通过 contact sheet sanity review，但仍建议在论文图中逐帧确认语义是否自然。 LPIPS 略差，作为定性图时需搭配 PSNR/SSIM 说明。 | dPSNR 3.3585, dMaskPSNR 3.3585, dLPIPS 0.0003 |
| 8 | YouTubeVOS100 | `990d358980` | top YouTubeVOS gain candidate: dPSNR=2.912, dSSIM=0.0042, dLPIPS=-0.0015 | SFT-48000 在 person/animal foreground 场景的 mask 区域容易出现局部模糊、贴片感或边界过渡不自然。 | Exp11 在该视频上 whole/mask PSNR 均提升 2.9119，说明外边界加权对补洞区域和上下文连续性有帮助。 | 该样本已通过 contact sheet sanity review，但仍建议在论文图中逐帧确认语义是否自然。 | dPSNR 2.9119, dMaskPSNR 2.9119, dLPIPS -0.0015 |
| 9 | YouTubeVOS100 | `860c0a7cf8` | top YouTubeVOS gain candidate: dPSNR=2.237, dSSIM=0.0033, dLPIPS=-0.0047 | SFT-48000 在 urban people / street 场景的 mask 区域容易出现局部模糊、贴片感或边界过渡不自然。 | Exp11 在该视频上 whole/mask PSNR 均提升 2.2368，说明外边界加权对补洞区域和上下文连续性有帮助。 | 该样本已通过 contact sheet sanity review，但仍建议在论文图中逐帧确认语义是否自然。 | dPSNR 2.2368, dMaskPSNR 2.2368, dLPIPS -0.0047 |
| 10 | YouTubeVOS100 | `1e458b1539` | top YouTubeVOS gain candidate: dPSNR=1.956, dSSIM=0.0035, dLPIPS=-0.0008 | SFT-48000 在 person / water 场景的 mask 区域容易出现局部模糊、贴片感或边界过渡不自然。 | Exp11 在该视频上 whole/mask PSNR 均提升 1.9564，说明外边界加权对补洞区域和上下文连续性有帮助。 | 该样本已通过 contact sheet sanity review，但仍建议在论文图中逐帧确认语义是否自然。 | dPSNR 1.9564, dMaskPSNR 1.9564, dLPIPS -0.0008 |
| 11 | YouTubeVOS100 | `c5b94822e3` | top YouTubeVOS gain candidate: dPSNR=1.848, dSSIM=0.0086, dLPIPS=-0.0007 | SFT-48000 在 close-up texture / object 场景的 mask 区域容易出现局部模糊、贴片感或边界过渡不自然。 | Exp11 在该视频上 whole/mask PSNR 均提升 1.8475，说明外边界加权对补洞区域和上下文连续性有帮助。 | 该样本已通过 contact sheet sanity review，但仍建议在论文图中逐帧确认语义是否自然。 | dPSNR 1.8475, dMaskPSNR 1.8475, dLPIPS -0.0007 |
| 12 | YouTubeVOS100 | `b0313efe37` | top YouTubeVOS gain candidate: dPSNR=1.781, dSSIM=0.0065, dLPIPS=-0.0030 | SFT-48000 在 vehicle / street 场景的 mask 区域容易出现局部模糊、贴片感或边界过渡不自然。 | Exp11 在该视频上 whole/mask PSNR 均提升 1.7815，说明外边界加权对补洞区域和上下文连续性有帮助。 | 该样本已通过 contact sheet sanity review，但仍建议在论文图中逐帧确认语义是否自然。 | dPSNR 1.7815, dMaskPSNR 1.7815, dLPIPS -0.0030 |
| 13 | YouTubeVOS100 | `3e2336812c` | top YouTubeVOS gain candidate: dPSNR=1.682, dSSIM=0.0014, dLPIPS=-0.0005 | SFT-48000 在 large motion / landscape 场景的 mask 区域容易出现局部模糊、贴片感或边界过渡不自然。 | Exp11 在该视频上 whole/mask PSNR 均提升 1.6819，说明外边界加权对补洞区域和上下文连续性有帮助。 | 该样本已通过 contact sheet sanity review，但仍建议在论文图中逐帧确认语义是否自然。 | dPSNR 1.6819, dMaskPSNR 1.6819, dLPIPS -0.0005 |
| 14 | YouTubeVOS100 | `eda3a7bbb1` | top YouTubeVOS gain candidate: dPSNR=1.612, dSSIM=0.0120, dLPIPS=-0.0034 | SFT-48000 在 underwater / background texture 场景的 mask 区域容易出现局部模糊、贴片感或边界过渡不自然。 | Exp11 在该视频上 whole/mask PSNR 均提升 1.6117，说明外边界加权对补洞区域和上下文连续性有帮助。 | 该样本已通过 contact sheet sanity review，但仍建议在论文图中逐帧确认语义是否自然。 | dPSNR 1.6117, dMaskPSNR 1.6117, dLPIPS -0.0034 |
| 15 | YouTubeVOS100 | `af881cd801` | top YouTubeVOS gain candidate: dPSNR=1.645, dSSIM=0.0056, dLPIPS=-0.0010 | SFT-48000 在 plant / thin structure 场景的 mask 区域容易出现局部模糊、贴片感或边界过渡不自然。 | Exp11 在该视频上 whole/mask PSNR 均提升 1.6448，说明外边界加权对补洞区域和上下文连续性有帮助。 | 该样本已通过 contact sheet sanity review，但仍建议在论文图中逐帧确认语义是否自然。 | dPSNR 1.6448, dMaskPSNR 1.6448, dLPIPS -0.0010 |
| 16 | YouTubeVOS100 | `f00dc892b2` | top YouTubeVOS gain candidate: dPSNR=1.467, dSSIM=0.0051, dLPIPS=0.0031 | SFT-48000 在 person / low-light indoor 场景的 mask 区域容易出现局部模糊、贴片感或边界过渡不自然。 | Exp11 在该视频上 whole/mask PSNR 均提升 1.4670，说明外边界加权对补洞区域和上下文连续性有帮助。 | 该样本已通过 contact sheet sanity review，但仍建议在论文图中逐帧确认语义是否自然。 LPIPS 略差，作为定性图时需搭配 PSNR/SSIM 说明。 | dPSNR 1.4670, dMaskPSNR 1.4670, dLPIPS 0.0031 |
| 17 | YouTubeVOS100 | `966c4c022e` | top YouTubeVOS gain candidate: dPSNR=1.258, dSSIM=0.0051, dLPIPS=-0.0025 | SFT-48000 在 animal / occlusion 场景的 mask 区域容易出现局部模糊、贴片感或边界过渡不自然。 | Exp11 在该视频上 whole/mask PSNR 均提升 1.2583，说明外边界加权对补洞区域和上下文连续性有帮助。 | 该样本已通过 contact sheet sanity review，但仍建议在论文图中逐帧确认语义是否自然。 | dPSNR 1.2583, dMaskPSNR 1.2583, dLPIPS -0.0025 |
| 18 | YouTubeVOS100 | `dcd3e1b53e` | top YouTubeVOS gain candidate: dPSNR=1.292, dSSIM=0.0004, dLPIPS=-0.0003 | SFT-48000 在 person / snow motion 场景的 mask 区域容易出现局部模糊、贴片感或边界过渡不自然。 | Exp11 在该视频上 whole/mask PSNR 均提升 1.2923，说明外边界加权对补洞区域和上下文连续性有帮助。 | 该样本已通过 contact sheet sanity review，但仍建议在论文图中逐帧确认语义是否自然。 | dPSNR 1.2923, dMaskPSNR 1.2923, dLPIPS -0.0003 |
| 19 | YouTubeVOS100 | `e0daa3b339` | top YouTubeVOS gain candidate: dPSNR=1.238, dSSIM=0.0052, dLPIPS=-0.0011 | SFT-48000 在 person / outdoor motion 场景的 mask 区域容易出现局部模糊、贴片感或边界过渡不自然。 | Exp11 在该视频上 whole/mask PSNR 均提升 1.2375，说明外边界加权对补洞区域和上下文连续性有帮助。 | 该样本已通过 contact sheet sanity review，但仍建议在论文图中逐帧确认语义是否自然。 | dPSNR 1.2375, dMaskPSNR 1.2375, dLPIPS -0.0011 |
| 20 | YouTubeVOS100 | `4c269afea9` | top YouTubeVOS gain candidate: dPSNR=1.246, dSSIM=0.0030, dLPIPS=-0.0014 | SFT-48000 在 water / long background 场景的 mask 区域容易出现局部模糊、贴片感或边界过渡不自然。 | Exp11 在该视频上 whole/mask PSNR 均提升 1.2457，说明外边界加权对补洞区域和上下文连续性有帮助。 | 该样本已通过 contact sheet sanity review，但仍建议在论文图中逐帧确认语义是否自然。 | dPSNR 1.2457, dMaskPSNR 1.2457, dLPIPS -0.0014 |


## DPO Diagnostics

Diagnostics snapshots:

```text
/home/hj/dpo-2-1-exp/this_week_exp11_exp12/dpo_diag/exp11_boundary_outer_b075_s1_dpo_diagnostics.csv
/home/hj/dpo-2-1-exp/this_week_exp11_exp12/dpo_diag/exp11_boundary_outer_b075_s2_dpo_diagnostics.csv
```

Summary report:

```text
reports/exp11_outer_b075_s2_dpo_diag_summary.md
```

Last-20 diagnostic means:

| Stage | dpo_loss | implicit_acc | raw_win_gap | raw_lose_gap | norm_win_gap | norm_lose_gap | norm_lose_gap_clipped | winner_abs_reg | winner_gap_reg | mse_w/ref | mse_l/ref | sigma_term | kl_divergence | loser_dominant_ratio | grad_norm | boundary_mode | boundary_weight | outside_weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Stage1 | 0.310162 | 0.987500 | -0.000332 | 0.550082 | -0.015532 | 1.213880 | 0.829626 | 0.128041 | 0.008638 | 0.987136 | 3.958190 | 0.747539 | 0.137437 | 1.000000 | 58.189300 | outer | 0.75 | 0.05 |
| Stage2 | 0.338639 | 1.000000 | 0.000083 | 0.513343 | -0.000614 | 1.208410 | 0.787290 | 0.090691 | 0.006708 | 0.999107 | 4.114820 | 0.721619 | 0.128356 | 1.000000 | 4.303000 | outer | 0.75 | 0.05 |

Conclusion: Exp11 outer b0.75 S2 is the current best result, but it must be reported with metric + visual + dpo-diag together. The diagnostics do not show old-style raw-DPO win-gap explosion, but they still show a loser-dominant pattern, so the method should not be sold using score alone.

## Exp12 Interpretation

Exp12 adaptive normalization tested a stronger normalization idea beyond Exp9/Exp10:

- log-ratio is reference-level normalization: it turns raw error difference into error ratio relative to the frozen reference.
- adaptive normalization is batch/timestep-level normalization: it tries to normalize the distribution of gaps within a batch or timestep group.

Current result: Exp12 did not beat Exp11 outer b0.75 S2, so it remains an ablation / negative comparison. It is useful for explaining that adaptive normalization alone does not replace boundary-aware region weighting.
