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

## 2026-06-29 PSNR-Safe SFT 30-Step Grid

All recipe aggregates are search-negative, so no 100-step or DPO-after-SFT gate
is unlocked.

| recipe | full dB | mask dB | boundary dB | outside dB | temporal proxy | full-positive rows | safe-positive rows | large-artifact rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `SFTmC_S0_lr3em05` | -1.816781 | -1.634597 | -1.899575 | -2.624405 | 0.525111 | 5 | 1 | 2 |
| `SFTmA_S0_lr3em05` | -2.303278 | -1.711663 | -1.975436 | -3.365938 | 0.584767 | 5 | 1 | 3 |
| `SFTmB_S0_lr3em05` | -2.306074 | -1.666656 | -1.952630 | -3.355257 | 0.595906 | 5 | 1 | 3 |
| `SFTmD_S0_lr3em05` | -2.395389 | -2.186769 | -2.440809 | -3.140433 | 0.414593 | 5 | 0 | 4 |
| `SFTmC_S0_lr0.0001` | -6.736656 | -5.878329 | -5.605923 | -8.168581 | 0.938385 | 0 | 0 | 16 |
| `SFTmD_S0_lr0.0001` | -6.801210 | -6.469689 | -6.300287 | -7.698793 | 1.156952 | 1 | 0 | 14 |
| `SFTmB_S0_lr0.0001` | -6.963161 | -5.897037 | -5.718433 | -8.264182 | 0.436655 | 0 | 0 | 15 |
| `SFTmA_S0_lr0.0001` | -7.172223 | -6.615492 | -6.301120 | -8.013239 | 0.475373 | 0 | 0 | 17 |
| `SFTmB_S0_lr0.0003` | -10.849289 | -9.839992 | -9.675967 | -12.674501 | 7.373321 | 0 | 0 | 24 |
| `SFTmA_S0_lr0.0003` | -13.827806 | -10.202384 | -10.424003 | -16.048283 | 7.137381 | 0 | 0 | 24 |
| `SFTmC_S0_lr0.0003` | -14.573955 | -10.318876 | -11.124060 | -17.201737 | 28.548663 | 0 | 0 | 24 |
| `SFTmD_S0_lr0.0003` | -15.079110 | -10.781672 | -11.678147 | -17.713054 | 24.738019 | 0 | 0 | 24 |

Metric note: LPIPS/Ewarp are not produced by the current isolated MiniMax
runner. No substitute values are inferred.
