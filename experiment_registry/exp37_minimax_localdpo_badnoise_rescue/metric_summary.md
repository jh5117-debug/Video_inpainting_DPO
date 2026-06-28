# Exp37 Metric Summary

No Exp37 metrics have been generated yet.

Readback imported prior MiniMax metric status:

- Exp30 frozen/EMA 10-step heldout metrics were near-tie/slightly negative.
- Exp35 R1/R2/R3 hard-noise rescue had negative mask/boundary/outside mean
  deltas.
- Exp36 winner-SFT best local movement was S1 LR `1e-5` with mask PSNR
  `+0.000986` and boundary PSNR `-0.004270`, not visually meaningful.

## 2026-06-28 Train-vs-Heldout Diagnosis

Status: `MINIMAX_OBJECTIVE_SIGNAL_TOO_WEAK`.

| Split | Rows | Full PSNR Delta | Mask PSNR Delta | Boundary PSNR Delta | Outside PSNR Delta | Mask Delta > 0 | Boundary Delta > 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train16 | 16 | +0.029083 | -0.008362 | +0.001128 | +0.048654 | 9/16 | 7/16 |
| heldout16 | 16 | -0.010218 | -0.008293 | -0.010939 | -0.014499 | 6/16 | 5/16 |

The numeric movement is small and not visually meaningful. Train-side local
mask PSNR is still negative on average, and heldout is negative across full,
mask, boundary, and outside PSNR.

## 2026-06-28 LocalDPO-style OR Corruption Pool

Status: `LOCALDPO_STYLE_POOL_READY_VISUAL_REVIEW_PASS`.

| Split | Rows | Auto Usable | Codex Final Usable | Final Medium-Hard | Final Hard-Plausible | Mean Outside MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train32 | 32 | 25 | 32 | 24 | 8 | 0.831835 |
| heldout16 | 16 | 14 | 16 | 14 | 2 | 0.260967 |
| total | 48 | 39 | 48 | 38 | 10 | - |

The auto rule preserved in the CSV was intentionally conservative. The final
classification keeps `auto_classification` and writes Codex-reviewed final
classification separately.
