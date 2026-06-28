# Exp40 Metric Summary

No Exp40 model outputs have been generated yet.

Readback imported Exp38 R1 heldout13 metrics:

- full PSNR delta: `+0.102167`
- mask PSNR delta: `+0.117230`
- boundary PSNR delta: `-0.141510`
- outside PSNR delta: `-0.037262`
- full-positive rows: `7/13`
- mask-positive rows: `7/13`
- boundary-negative rows: `9/13`
- outside-negative rows: `8/13`
- outside-MAE-worse rows: `10/13`

This is not a positive gate. Exp40 must exceed `+0.2 dB` on shadow while making
boundary/outside safe and preserving LPIPS/Ewarp.

## 2026-06-28 R1 Sample-Level Diagnosis

Available existing evidence:

- Exp38 SFT/DPO R1 heldout13: full/mask/boundary/outside means
  `+0.102167` / `+0.117230` / `-0.141510` / `-0.037262`.
- Exp38 train-overfit Exp37 R1 train32: full/mask/boundary/outside means
  `-0.586255` / `+0.152062` / `+0.069123` / `-0.895018`.
- Exp38 train-overfit Exp37 R1 heldout16: full/mask/boundary/outside means
  `+0.200826` / `+0.161946` / `-0.049755` / `+0.028198`.

The train32 evidence shows mask/boundary motion but full/outside regression;
the heldout evidence shows the desired full/mask direction but boundary is not
safe. This points to PSNR-safe SFT with stronger outside and boundary
preservation before any DPO-after-SFT gate.

## 2026-06-29 LocalDPO v3 Pool Metrics

Pool construction metrics, not model metrics:

- candidate rows: `336`
- selected rows: `112`
- selected split counts: `train=64`, `search=24`, `shadow=24`
- selected classification: `112/112 MEDIUM_HARD_ELIGIBLE`
- rejected rows: `46`
- candidate bucket counts:
  - `MEDIUM_HARD_ELIGIBLE=282`
  - `HARD_BUT_PLAUSIBLE=9`
  - `TOO_CLOSE=14`
  - `TRIVIAL_BAD=31`
- scene overlap:
  - train/search: `0`
  - train/shadow: `0`
  - search/shadow: `0`
- source balance:
  - train: `BLENDER=32`, `REAL=32`
  - search: `BLENDER=12`, `REAL=12`
  - shadow: `BLENDER=12`, `REAL=12`

The full requested pool size was not reached; the pre-registered minimum was
reached. Step0/SFT diagnostics may use this pool with the minimum-pool caveat.

## 2026-06-29 Step0 Baseline Metrics

Raw MiniMax Step0 baseline, inference only:

| split | rows | full PSNR | mask PSNR | boundary PSNR | outside PSNR | outside MAE | temporal diff MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 64 | 23.965598 | 18.485359 | 19.395954 | 26.458319 | 10.077378 | 2.334655 |
| search | 24 | 25.043807 | 20.493872 | 21.409812 | 27.765446 | 8.877102 | 3.845744 |
| shadow | 24 | 26.209732 | 21.645338 | 24.277694 | 29.577002 | 7.695790 | 2.199505 |

No training or checkpoint selection occurred. These numbers are the fixed
baseline for PSNR-safe SFT and any later DPO-after-SFT comparisons.
