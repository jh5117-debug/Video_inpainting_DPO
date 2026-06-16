# PRD 31: OR150 Object-Removal Benchmark

Date: 2026-06-16

## Goal

Move from the current BR/inpainting story into an object-removal benchmark that
uses true foreground masks where available.

The target comparison requested for paper/PPT is:

```text
MiniMax-Remover
COCOCO
FloED
DiffuEraser SFT-48000
VideoPainter
VACE
VideoComp / VideoComposer
DiffuEraser Exp11 outer b0.75 S2
```

This is frozen inference/evaluation only. It is not adapter training.

## Dataset

### DAVIS50 OR

Use the same 50 video names as the current DAVIS50 protocol, but take masks from
DAVIS2017 foreground annotations:

```text
HAL: /home/hj/Video_inpainting_DPO/data/external/davis_2017_full_resolution/DAVIS
PAI: /mnt/nas/hj/data/external/davis_2017_full_resolution_or_eval50/DAVIS
```

Mask semantics:

```text
nonzero DAVIS2017 annotation = foreground object to remove.
```

This is a different evaluation task from the previous BR/DAVIS432 masks.

### YouTubeVOS100 OR

Use the fixed YouTubeVOS100 eval subset already staged on PAI:

```text
/mnt/workspace/hj/nas_hj/data/external/youtubevos_432_240_eval100
```

Manifest:

```text
exp15_or_benchmark/manifests/youtubevos100_or_manifest.csv
```

### Combined Manifest

```text
exp15_or_benchmark/manifests/or150_manifest.csv
```

Counts:

| Split | Videos |
|---|---:|
| DAVIS50 OR | 50 |
| YouTubeVOS100 OR | 100 |
| OR150 total | 150 |

## Existing Best Context

Current best BR/inpainting result:

| Dataset | Method | PSNR | SSIM | LPIPS | VFID | TC |
|---|---|---:|---:|---:|---:|---:|
| DAVIS50 | SFT-48000 | 32.7314 | 0.9705 | 0.0167 | 0.2018 | 0.9712 |
| DAVIS50 | Exp11 outer b0.75 S2 | 33.0140 | 0.9723 | 0.0154 | 0.1754 | 0.9711 |
| YouTubeVOS100 | SFT-48000 | 33.3968 | 0.9701 | 0.0176 | 0.2007 | 0.9819 |
| YouTubeVOS100 | Exp11 outer b0.75 S2 | 33.7238 | 0.9711 | 0.0168 | 0.1925 | 0.9821 |

These numbers remain useful context but are not yet the true OR150 table for
DAVIS2017 foreground masks.

## Runtime Readiness

| Method | OR150 Status | Notes |
|---|---|---|
| DiffuEraser SFT-48000 | ready | Use project OR wrapper and SFT-48000 weights. |
| DiffuEraser Exp11 outer b0.75 S2 | ready with rerun | Must rerun on OR150 true masks before claiming OR result. |
| MiniMax-Remover | ready frozen baseline, env pending | Real weights and official repo are present; needs isolated env with newer diffusers before inference. |
| COCOCO | ready after one-video env smoke | Real weights and official repo exist on PAI/NAS. |
| VideoPainter | ready frozen baseline | Use official inference setting and project metric wrapper. |
| FloED | blocked | No verified local/PAI repo+weights+OR wrapper yet. |
| VACE | blocked | No verified local/PAI repo+weights+OR wrapper yet. |
| VideoComp / VideoComposer | blocked | No clean OR-compatible runtime validated. |

## Visualization

The desired method comparison is 8 method outputs:

1. MiniMax-Remover
2. COCOCO
3. FloED
4. DiffuEraser SFT-48000
5. VideoPainter
6. VACE
7. VideoComp / VideoComposer
8. DiffuEraser Exp11 outer b0.75 S2

GT/original and mask overlay should still be saved for every case. For PPT,
either place them as auxiliary columns/rows or make a separate source/mask
panel. If the final figure must be exactly 8 visual columns, use the 8 method
outputs above and put GT/mask in the caption or a separate row. Do not include
fake outputs for blocked methods.

## Metric Protocol

- Use project `inference/metrics.py` / metric wrapper.
- No VBench.
- Do not compute metrics through mp4 round-trip.
- Hard comp for metric frames:

```text
comp = prediction_inside_mask + GT_outside_mask
```

- No mask dilation during evaluation unless a baseline's own official inference
  requires a preprocessing mask; if it does, record it as method-specific
  inference setting.

## Storage Rule

HAL is only a transit node for large weights. Download to HAL only when PAI
cannot download directly, rsync to PAI/NAS, then remove HAL temporary weights if
space becomes tight.

Do not commit weights, videos, generated frames, or full metric outputs.

## Next Gate

Run one-video OR smoke on PAI for the methods currently ready:

```text
DiffuEraser SFT-48000
DiffuEraser Exp11 outer b0.75 S2
COCOCO
VideoPainter
MiniMax-Remover after isolated env is built
```

If smoke passes, launch full OR150 inference/eval for those ready methods.

Known PAI repo paths for frozen baselines:

```text
MiniMax-Remover: /mnt/nas/hj/official_repos/MiniMax-Remover_28e12b4
COCOCO: /mnt/nas/hj/official_repos/COCOCO_9ebe984
VideoPainter: /mnt/nas/hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter
```
