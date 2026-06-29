# Exp43 Metric Summary

No Exp43 training or evaluation metric has been run yet.

## 2026-06-29 H20 GPU Release

Runtime readiness only. No model-quality metric was produced.
# Exp43 Metric Summary

Current status: `H20_EXP43_SFT_BLOCKED`.

## 2026-06-29 SFT-A 30-Step Cell

Run: `SFT-A_lr3em5_step30`

| split | rows | dPSNR | dMask PSNR | dBoundary PSNR | dOutside PSNR | dEwarp | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| search | 24 | -5.8331 | -4.6745 | -4.7009 | -7.5941 | +0.6461 | FAIL |
| shadow | 24 | -6.5506 | -4.2232 | -5.3735 | -8.4532 | +0.5934 | FAIL |

LPIPS remains `LPIPS_RUNTIME_BLOCKED`; the initial missing `torchmetrics`
dependency was repaired in an Exp43 runtime-local dependency directory, but the
reuse-existing LPIPS rerun exceeded reasonable runtime and was stopped.

No 100-step, 300-step, DPO-after-SFT, or 500-step confirmation was unlocked.

No Exp43 training metrics exist yet. Readback reuses prior gates only as
prerequisite evidence:

- Exp41 H20 data audit: `2242` active refs checked, `0` missing.
- Exp41 H20 LocalDPO v3 pool: `train64/search24/shadow24`.
- Exp41 BF16 preflight: P0-P7 passed, including DDP8 one-batch training.
- Exp41 official protocol audit: executable MiniMax protocol matched.

## 2026-06-29 BF16 Preflight

P0-P7 passed. Rank0 summary:

| case | loss | grad norm | peak MiB | status |
| --- | ---: | ---: | ---: | --- |
| P0 | 2050.82861328125 | 2.831369161605835 | 132.0 | PASS |
| P1 |  |  | 6260.286 | PASS |
| P2 | 0.19614098966121674 | 0.0 | 7274.638 | PASS |
| P3 | 0.19614098966121674 | 3.281572142587461 | 59932.281 | PASS |
| P4 | 0.19492636620998383 | 3.179090593773888 | 68982.773 | PASS |
| P5 | 0.19614098966121674 | 3.281572142587461 | 59932.281 | PASS |
| P6 | 0.19614098966121674 | 2.1718841431594367 | 62087.76 | PASS |
| P7 | 0.19614098966121674 | 1.488086613488801 | 62087.76 | PASS |

This unlocks data readiness and gated SFT ladder work only.

## 2026-06-29 Data Readiness

| split | rows | scene groups | BLENDER | REAL | required failures | VOR-Eval | hard comp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 64 | 64 | 32 | 32 | 0 | 0 | 0 |
| search | 24 | 24 | 12 | 12 | 0 | 0 | 0 |
| shadow | 24 | 24 | 12 | 12 | 0 | 0 | 0 |

Full `train96/search32/shadow32` is not available.
