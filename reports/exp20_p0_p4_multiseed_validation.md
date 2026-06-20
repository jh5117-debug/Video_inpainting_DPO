# Exp20 P0/P4 Multi-Seed Validation

Source files:

- `reports/exp20_first_wave_full_metrics.csv` for seed 20260619.
- `reports/exp20_p0_p4_multiseed_full_metrics.csv` for seeds 20260620 and 20260621.

## Three-Seed Summary

| trial | PSNR mean | PSNR std | SSIM mean | LPIPS mean | TC mean | Ewarp mean | VFID/FVD | mask PSNR | boundary PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | 29.362203 | 0.012115 | 0.968835 | 0.017952 | 0.976110 | 12.016635 | 0.225063 | 17.056892 | 22.935946 |
| P4 | 29.382922 | 0.012659 | 0.968921 | 0.018256 | 0.975991 | 11.991874 | 0.229764 | 17.077611 | 22.949209 |

## Paired P4 - P0 Deltas

| seed | delta PSNR | delta SSIM | delta LPIPS | delta TC | delta Ewarp | delta VFID/FVD | delta mask PSNR | delta boundary PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260619 | +0.016701 | +0.000084 | +0.000356 | -0.000091 | -0.021972 | +0.008453 | +0.016701 | -0.000119 |
| 20260620 | +0.018639 | +0.000053 | +0.000248 | -0.000117 | -0.026996 | +0.003222 | +0.018639 | +0.021979 |
| 20260621 | +0.026817 | +0.000120 | +0.000309 | -0.000149 | -0.025317 | +0.002428 | +0.026817 | +0.017929 |

## Gate

- P4 beats P0 in `3/3` seeds.
- Mean paired PSNR delta P4-P0: `+0.020719` dB.
- P4 mean PSNR minus Exp11-S1 dev baseline: `+0.049381` dB.
- P4 stability gate: `PASS`.

Interpretation: P4 remains a small but repeatable dev-pilot signal. It is not a final method and must still survive fixed/RB/adaptive search, equal-step, shadow-dev, and visual checks.
