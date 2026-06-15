# PRD 23: YouTubeVOS100 + DAVIS50 Extended Eval

Date: 2026-06-15

## Goal

Evaluate the current best method beyond DAVIS50:

- SFT-48000 DiffuEraser baseline
- Exp11 boundary outer b0.75 S2

Datasets:

- DAVIS50
- YouTubeVOS100

## Fixed Protocol

This PRD uses the same fixed video inpainting protocol as the final DAVIS table:

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

Do not use VBench for these inpainting results.

## YouTubeVOS100 Dataset

HAL source:

```text
/home/hj/Video_inpainting_DPO/data/external/youtubevos_432_240
```

PAI target:

```text
/mnt/workspace/hj/nas_hj/data/external/youtubevos_432_240_eval100
```

Sample manifest:

```text
/home/hj/dpo-2-1-exp/this_week_exp11_exp12/youtubevos100/sample_manifest.csv
```

Sampling:

- fixed seed: `20260615`
- eligible videos: videos with at least 24 common image/mask frames
- selected videos: 100
- directory structure preserved

## Eval Outputs

PAI YouTubeVOS100 output:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp11_outer_b075_s2_youtubevos100_20260615_194218_youtubevos100_raw6_hardcomp
```

HAL YouTubeVOS100 snapshot:

```text
/home/hj/dpo-2-1-exp/this_week_exp11_exp12/youtubevos100_eval/exp11_outer_b075_s2_youtubevos100_20260615_194218_youtubevos100_raw6_hardcomp
```

DAVIS50 metrics source:

```text
/home/hj/dpo-2-1-exp/this_week_exp11_exp12/metrics/summary_all.csv
```

Consolidated report:

```text
reports/exp11_outer_b075_s2_youtubevos100_davis50_eval.md
reports/exp11_outer_b075_s2_youtubevos100_davis50_eval.csv
```

## Main Table

| Dataset | Method | Rows | PSNR | SSIM | LPIPS | VFID | TC | Mask PSNR | Mask SSIM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DAVIS50 | SFT-48000 baseline | 50 | 32.7314 | 0.9705 | 0.0167 | 0.2018 | 0.9712 | 23.8849 | 0.7976 |
| DAVIS50 | Exp11 outer b0.75 S2 | 50 | 33.0140 | 0.9723 | 0.0154 | 0.1754 | 0.9711 | 24.1675 | 0.8099 |
| YouTubeVOS100 | SFT-48000 baseline | 100 | 33.3968 | 0.9701 | 0.0176 | 0.2007 | 0.9819 | 24.4262 | 0.7935 |
| YouTubeVOS100 | Exp11 outer b0.75 S2 | 100 | 33.7238 | 0.9711 | 0.0168 | 0.1925 | 0.9821 | 24.7532 | 0.7990 |

## Deltas

| Dataset | PSNR Delta | SSIM Delta | LPIPS Delta | VFID Delta | TC Delta | Mask PSNR Delta |
|---|---:|---:|---:|---:|---:|---:|
| DAVIS50 | +0.2826 | +0.0018 | -0.0013 | -0.0264 | -0.0001 | +0.2826 |
| YouTubeVOS100 | +0.3270 | +0.0009 | -0.0008 | -0.0083 | +0.0001 | +0.3270 |

## Interpretation

Exp11 outer b0.75 S2 improves over SFT-48000 on both DAVIS50 and YouTubeVOS100 under the fixed raw6 hard-comp protocol.

The improvement is not only whole-frame: mask PSNR also improves on both datasets. This matters because hard comp can make whole-frame PSNR too forgiving outside the hole region.

Ewarp is not present in the current consolidated summary. The available table reports PSNR, SSIM, LPIPS, VFID, TC, mask PSNR, and mask SSIM. If Ewarp is needed for the paper table, it should be computed as an additional metric-only pass without changing the protocol or rerunning training.

## Link To Visual Evidence

Final 20 paper/PPT-ready cases are archived at:

```text
/home/hj/dpo-2-1-exp/final_20_visual_cases_for_paper
```

The package combines DAVIS50 and YouTubeVOS100 cases and includes four-column videos, contact sheets, frame-by-frame panels, and per-video metrics.
