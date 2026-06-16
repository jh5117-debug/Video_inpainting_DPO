# OR150 Benchmark Initial Plan

Date: 2026-06-16

## Goal

Evaluate object-removal baselines on the same 150-video pool used for the
current DiffuEraser Exp11 outer b0.75 S2 result:

- DAVIS50 with real DAVIS2017 foreground masks;
- YouTubeVOS100 fixed-seed subset;
- frozen baselines only;
- no adapter training.

## Current Best Reference

Existing DiffuEraser Exp11 outer b0.75 S2 numbers:

| Dataset | Method | PSNR | SSIM | LPIPS | VFID | TC |
|---|---|---:|---:|---:|---:|---:|
| DAVIS50 | SFT-48000 | 32.7314 | 0.9705 | 0.0167 | 0.2018 | 0.9712 |
| DAVIS50 | Exp11 outer b0.75 S2 | 33.0140 | 0.9723 | 0.0154 | 0.1754 | 0.9711 |
| YouTubeVOS100 | SFT-48000 | 33.3968 | 0.9701 | 0.0176 | 0.2007 | 0.9819 |
| YouTubeVOS100 | Exp11 outer b0.75 S2 | 33.7238 | 0.9711 | 0.0168 | 0.1925 | 0.9821 |

These are BR/inpainting metrics with the current masks. For OR, DAVIS masks must
come from DAVIS2017 foreground annotations.

## Data Decision

DAVIS2017 full-resolution exists on HAL:

```text
/home/hj/Video_inpainting_DPO/data/external/davis_2017_full_resolution/DAVIS
```

It contains 90 videos. The OR benchmark uses the same 50 video names as the
current DAVIS50 protocol, with real foreground annotations from:

```text
Annotations/Full-Resolution/<video>
```

PAI target:

```text
/mnt/nas/hj/data/external/davis_2017_full_resolution_or_eval50/DAVIS
```

Workspace symlink:

```text
/mnt/workspace/hj/nas_hj/data/external/davis_2017_full_resolution_or_eval50
```

YouTubeVOS100 already exists on PAI:

```text
/mnt/workspace/hj/nas_hj/data/external/youtubevos_432_240_eval100
```

## Baseline Scope

Requested comparison:

```text
MiniMax-Remover
CoCoCo
FloED
DiffuEraser SFT-48000
VideoPainter
VACE
VideoComp / VideoComposer if runnable
DiffuEraser Exp11 outer b0.75 S2
```

Important caveat:

Some methods have public inference code but no validated local weights or no
usable repo. A method must have a runnable inference path before it can enter the
OR150 table. Otherwise it is marked blocked instead of being faked.

## Storage Rule

HAL has enough temporary space but should be treated as a transit node only.
Large weights should be downloaded to HAL only if PAI cannot download them, then
rsynced to PAI/NAS and removed from HAL cache if needed.

Do not commit weights, datasets, videos, or generated frames.
