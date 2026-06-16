# OR150 Dataset Transfer Audit

Date: 2026-06-16

## Scope

This audit prepares the object-removal benchmark requested for the paper table.
It is not an adapter experiment and does not start any training.

The benchmark pool is:

- DAVIS50 OR: the same 50 DAVIS video names used by the fixed DAVIS50 protocol,
  but with true DAVIS2017 foreground annotation masks.
- YouTubeVOS100 OR: the existing fixed-seed YouTubeVOS100 eval subset already
  staged on PAI.

## DAVIS50 OR

HAL source:

```text
/home/hj/Video_inpainting_DPO/data/external/davis_2017_full_resolution/DAVIS
```

PAI/NAS target:

```text
/mnt/nas/hj/data/external/davis_2017_full_resolution_or_eval50/DAVIS
```

PAI verification:

```text
host = dsw-753014-dc85766cb-4v2jj
size = 1.5G
JPEGImages/Full-Resolution video dirs = 50
Annotations/Full-Resolution mask dirs = 50
```

Mask semantics:

```text
DAVIS2017 foreground annotation; nonzero means object to remove.
```

This is different from the earlier BR/inpainting masks used for Exp9/10/11.
DAVIS OR tables must not reuse the old DAVIS BR masks.

## YouTubeVOS100 OR

PAI path:

```text
/mnt/workspace/hj/nas_hj/data/external/youtubevos_432_240_eval100
```

Manifest:

```text
exp15_or_benchmark/manifests/youtubevos100_or_manifest.csv
```

Mask semantics:

```text
existing YouTubeVOS test mask; nonzero means object to remove.
```

## Manifests

Local HAL:

```text
exp15_or_benchmark/manifests/davis50_or_manifest.csv
exp15_or_benchmark/manifests/youtubevos100_or_manifest.csv
exp15_or_benchmark/manifests/or150_manifest.csv
exp15_or_benchmark/manifests/or150_manifest_summary.json
```

PAI sync target:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/exp15_or_benchmark/manifests/
```

Counts:

| Split | Count |
|---|---:|
| DAVIS50 OR | 50 |
| YouTubeVOS100 OR | 100 |
| Total OR150 | 150 |

## Existing DiffuEraser Reference Numbers

These are the current BR/inpainting protocol numbers and should be treated as
reference context until the true OR masks are evaluated:

| Dataset | Method | PSNR | SSIM | LPIPS | VFID | TC |
|---|---|---:|---:|---:|---:|---:|
| DAVIS50 | SFT-48000 | 32.7314 | 0.9705 | 0.0167 | 0.2018 | 0.9712 |
| DAVIS50 | Exp11 outer b0.75 S2 | 33.0140 | 0.9723 | 0.0154 | 0.1754 | 0.9711 |
| YouTubeVOS100 | SFT-48000 | 33.3968 | 0.9701 | 0.0176 | 0.2007 | 0.9819 |
| YouTubeVOS100 | Exp11 outer b0.75 S2 | 33.7238 | 0.9711 | 0.0168 | 0.1925 | 0.9821 |

## Decision

OR150 data staging is ready. The next gate is baseline runtime readiness and
per-method inference, not dataset transfer.
