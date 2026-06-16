# PRD 31: OR150 Object-Removal Benchmark

Date: 2026-06-16

## 2026-06-16 Update: Reduced To DAVIS50 Only

The OR150 plan is paused for this stage. The active benchmark is now:

```text
Exp15 OR DAVIS50 only
```

This means:

- do not run YouTubeVOS100 in this gate;
- do not report OR150 numbers;
- use DAVIS2017 foreground object masks only;
- run full DAVIS50 for methods with a verified runtime;
- keep blocked methods explicit in tables and visual grids.

Current implementation folder:

```text
exp15_or_benchmark_davis50/
```

## Goal

Move from the current BR/inpainting story into an object-removal benchmark that
uses true foreground masks where available.

The target comparison requested for the current DAVIS50 OR benchmark is:

```text
ProPainter
VideoComp / VideoComposer
CoCoCo
FloED
DiffuEraser SFT-48000
VideoPainter
VACE
DiffuEraser Exp11 outer b0.75 S2
```

MiniMax-Remover is tracked as an extra method. If it becomes runnable, it can be
added to the auxiliary table and visual grids, but it should not block the main
8-method table.

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

Paused for this stage. The path below remains a future option, but it is not
used in the DAVIS50-only benchmark:

```text
/mnt/workspace/hj/nas_hj/data/external/youtubevos_432_240_eval100
```

Manifest:

```text
exp15_or_benchmark/manifests/youtubevos100_or_manifest.csv
```

### Combined Manifest

The OR150 combined manifest is historical/future context only. It is not the
active run.

Historical planned counts:

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

| Method | DAVIS50 OR Status | Notes |
|---|---|---|
| ProPainter | completed 50/50 | Existing wrapper and weights ran successfully. |
| DiffuEraser SFT-48000 | completed 50/50 | Exp15 isolated DiffuEraser OR wrapper, no comp metric. |
| DiffuEraser Exp11 outer b0.75 S2 | completed 50/50 | Same wrapper using Exp11 outer b0.75 S2 last weights. |
| CoCoCo | blocked | Repo/checkpoints exist but required SD inpainting dependency is incomplete. |
| VideoPainter | blocked | No verified DAVIS2017 foreground-mask OR wrapper. |
| FloED | blocked | No verified local/PAI repo+weights+OR wrapper. |
| VACE | blocked | No verified local/PAI repo+weights+OR wrapper. |
| VideoComp / VideoComposer | blocked | No verified local/PAI repo+weights+OR wrapper. |
| MiniMax-Remover | blocked extra | Requires isolated newer env; do not run in DiffuEraser env. |

## Visualization

The DAVIS50 visual grids use the requested 8 method slots:

1. ProPainter
2. VideoComp / VideoComposer
3. CoCoCo
4. FloED
5. DiffuEraser SFT-48000
6. VideoPainter
7. VACE
8. DiffuEraser Exp11 outer b0.75 S2

GT/original and mask overlay should still be saved for every case. For PPT,
either place them as auxiliary columns/rows or make a separate source/mask
panel. If the final figure must be exactly 8 visual columns, use the 8 method
outputs above and put GT/mask in the caption or a separate row. Do not include
fake outputs for blocked methods.

Generated Exp15 grids explicitly show blocked methods as unavailable
placeholders, rather than silently dropping them.

## Metric Protocol

- Use the Exp15 OR metric wrapper, which uses project image/video IO and keeps
  the metric definition explicit in `reports/exp15_or_metric_protocol.md`.
- No VBench.
- Do not compute metrics through mp4 round-trip.
- No hard comp for OR metric frames:

```text
input  = original frame
output = raw method output
bg     = mask == 0
```

- Primary metrics are `PSNR_bg`, `SSIM_bg_ignore_mask`, and `TC_bg` if
  available.
- Mask-inside quality is judged through visual comparison, because OR has no
  true removed-background GT inside the object mask.

## Storage Rule

HAL is only a transit node for large weights. Download to HAL only when PAI
cannot download directly, rsync to PAI/NAS, then remove HAL temporary weights if
space becomes tight.

Do not commit weights, videos, generated frames, or full metric outputs.

## Next Gate

Current DAVIS50 gate directly ran DAVIS50 for methods that were ready. No smoke
was used in this gate.

```text
ProPainter
DiffuEraser SFT-48000
DiffuEraser Exp11 outer b0.75 S2
```

Blocked methods remain explicit placeholders and must not be fabricated.

Known PAI repo paths for frozen baselines:

```text
MiniMax-Remover: /mnt/nas/hj/official_repos/MiniMax-Remover_28e12b4
COCOCO: /mnt/nas/hj/official_repos/COCOCO_9ebe984
VideoPainter: /mnt/nas/hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter
```
