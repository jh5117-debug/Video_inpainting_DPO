# PRD 32: OR DAVIS50 Benchmark Results

Date: 2026-06-16

## Scope

This PRD is for the active object-removal benchmark:

```text
Exp15 OR DAVIS50 only
```

It does not include YouTubeVOS100 and does not report OR150.

## Dataset

- Source: DAVIS2017 full-resolution foreground annotations.
- HAL source: `/home/hj/Video_inpainting_DPO/data/external/davis_2017_full_resolution/DAVIS`
- PAI target: `/mnt/nas/hj/data/external/davis_2017_full_resolution_or_eval50/DAVIS`
- Manifest: `exp15_or_benchmark_davis50/manifests/davis50_or_manifest.csv`

Mask semantics:

```text
mask != 0 = foreground object to remove
mask == 0 = original background to preserve
```

## Metric Protocol

OR metrics are not BR metrics.

- no comp before metric;
- raw method output is compared to the original frame;
- primary metric region is background only (`mask == 0`);
- `PSNR_bg` is strict background-pixel PSNR;
- `SSIM_bg_ignore_mask` is a background-preservation SSIM proxy;
- mask-inside removal quality is judged by visual evidence.

Protocol report:

```text
reports/exp15_or_metric_protocol.md
```

## Method Table

The active method list keeps all requested methods visible, but only verified
runtimes are executed.

| Method | Runtime status |
|---|---|
| ProPainter | COMPLETED_50_50 |
| VideoComposer / VideoComp | BLOCKED_NO_REPO |
| CoCoCo | BLOCKED_NO_WEIGHT |
| FloED | BLOCKED_NO_REPO |
| DiffuEraser SFT-48000 | COMPLETED_50_50 |
| VideoPainter | BLOCKED_NO_OR_WRAPPER |
| VACE | BLOCKED_NO_REPO |
| Ours Exp11 outer b0.75 S2 | COMPLETED_50_50 |
| MiniMax-Remover | BLOCKED_IMPORT_ERROR |

Runtime status report:

```text
reports/exp15_or_method_runtime_status.md
```

## Quantitative Result

Completed on PAI for the runnable methods.

Output:

```text
reports/exp15_or_davis50_fixed_quantitative_summary.csv
reports/exp15_or_davis50_fixed_quantitative_summary.md
```

| Method | Status | Success | PSNR_bg | SSIM_bg | TC_bg_pixel_proxy | Notes |
|---|---|---:|---:|---:|---:|---|
| ProPainter | ok | 50/50 | 35.5274 | 0.9927 | 35.7664 | Strongest background preservation among runnable methods. |
| VideoComposer / VideoComp | failed_or_blocked | 0/50 | n/a | n/a | n/a | No verified PAI repo+weights+OR wrapper. |
| CoCoCo | failed_or_blocked | 0/50 | n/a | n/a | n/a | Stable-diffusion-v1-5-inpainting dependency incomplete. |
| FloED | failed_or_blocked | 0/50 | n/a | n/a | n/a | No verified PAI repo+weights+OR wrapper. |
| DiffuEraser SFT-48000 | ok | 50/50 | 28.6773 | 0.9686 | 28.8505 | Runnable DiffuEraser baseline. |
| VideoPainter | failed_or_blocked | 0/50 | n/a | n/a | n/a | No verified DAVIS2017 OR wrapper. |
| VACE | failed_or_blocked | 0/50 | n/a | n/a | n/a | No verified PAI repo+weights+OR wrapper. |
| Ours Exp11 outer b0.75 S2 | ok | 50/50 | 28.6795 | 0.9685 | 28.8682 | Nearly tied with SFT on bg metrics; slightly higher PSNR_bg/TC_bg_pixel_proxy, slightly lower SSIM_bg. |
| MiniMax-Remover | failed_or_blocked | 0/50 | n/a | n/a | n/a | Needs isolated newer env; not run in shared DiffuEraser env. |

Interpretation:

- ProPainter is much stronger for this OR background-preservation protocol.
- Exp11 outer b0.75 S2 remains the best BR/DAVIS hard-comp configuration, but it is not an OR winner here.
- Ours is approximately tied with SFT-48000 on OR background metrics: +0.0022 PSNR_bg, -0.0002 SSIM_bg, +0.0177 TC_bg_pixel_proxy.

## Visual Result

Output:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp15_or_benchmark_davis50_fixed/visual_grids
reports/exp15_or_davis50_visual_case_report.md
reports/exp15_or_davis50_paper_ready_cases.md
```

Visual grids use two rows:

- row 1: input / mask / first method outputs;
- row 2: remaining method outputs and explicit BLOCKED placeholders.

HAL copy:

```text
/home/hj/dpo-2-1-exp/exp15_or_benchmark_davis50_visuals_fixed
```

The fixed visual grids replace the older OpenCV/mp4v outputs. The 20-case file is metric-preselected from DAVIS50 and should be treated as a
candidate list for paper/PPT figures. Final figure selection still needs human
visual review because OR mask-inside removal quality has no GT target.

## Current BR Context

The existing BR/hard-comp numbers for DiffuEraser are useful context but are
not this OR table:

| Dataset | Method | PSNR | SSIM | LPIPS | VFID | TC |
|---|---|---:|---:|---:|---:|---:|
| DAVIS50 | SFT-48000 | 32.7314 | 0.9705 | 0.0167 | 0.2018 | 0.9712 |
| DAVIS50 | Exp11 outer b0.75 S2 | 33.0140 | 0.9723 | 0.0154 | 0.1754 | 0.9711 |

Do not mix these BR values into the OR metric table.

## MiniMax Alignment Caveat

Current fixed Exp15 results are still DAVIS50-subset diagnostics. They are not a reproduction of MiniMax Table 2 because the paper uses DAVIS90, CLIP-feature TC, and GPT-O3 VQ/Succ. See `PRD/34_or_eval_protocol_minimax_alignment.md`.
