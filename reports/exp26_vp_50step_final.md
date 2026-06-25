# Exp26 VideoPainter 50-Step Gate

Status: `VIDEOPAINTER_ADAPTER_POSITIVE`

Scope: search-dev micro-training gate only. This is not `SCIENTIFIC_POSITIVE`, does not authorize RC-FPO, and does not authorize 100-step or longer training.

## Comp Metrics

| step | PSNR | SSIM | LPIPS | Ewarp | strict mask PSNR | boundary PSNR |
|---|---:|---:|---:|---:|---:|---:|
| step0 | 24.301897 | 0.871558 | 0.070801 | 8.042740 | 16.012427 | 16.011952 |
| step10 | 25.435825 | 0.903158 | 0.067957 | 6.790277 | 17.148051 | 21.176618 |
| step30 | 29.032479 | 0.956021 | 0.031781 | 1.956920 | 20.858997 | 25.278172 |
| step50 | 29.118066 | 0.959441 | 0.026741 | 0.987617 | 20.954673 | 28.123842 |

## Step50 Minus Step0

- PSNR: `+4.816168`
- SSIM: `+0.087883`
- LPIPS: `-0.044059`
- Ewarp: `-7.055122`
- strict mask PSNR: `+4.942246`
- boundary PSNR: `+12.111889`

## Paired Statistics

- PSNR mean delta: `+4.816168`
- PSNR per-video win rate: `0.718750`
- PSNR bootstrap 95% CI: `[+2.648960, +7.234666]`
- PSNR probability(delta > 0): `1.000000`
- PSNR leave-one-out range: `[+4.121639, +5.126012]`
- LPIPS mean delta: `-0.044059`, probability improved `1.000000`
- Ewarp mean delta: `-7.055122`, probability improved `1.000000`
- strict mask PSNR mean delta: `+4.942246`, probability improved `1.000000`
- boundary PSNR mean delta: `+12.111889`, probability improved `1.000000`

## Diagnostics

- diagnostic rows: `50`
- final total loss: `0.9509385228157043`
- final DPO loss: `0.8550137281417847`
- final implicit accuracy: `0.0`
- max grad norm: `471.68358081969296`
- p95 grad norm: `136.25990432375542`
- final loser-dominant ratio: `0.0`
- NaN/Inf count: `0`
- strict reload/preflight: checkpoint-10/20/30/40/50 all passed with zero missing/unexpected keys

## Manual Visual Review

- Reviewed all `32/32` step50 search-dev rows via 8 dense temporal evidence pages and 8 crop pages.
- No global black/purple collapse, no frame-order mismatch, no first-frame failure, no systematic outside damage, and no gate-blocking flicker/ghosting pattern was found.
- Remaining artifacts are local: green/purple residual patches, local blur, texture discontinuity, and boundary tinting in mask/affected regions.
- Strongest remaining failure cases include water/foliage/grass texture rows; these are retained as failure evidence, not hidden.

## Decision

The pre-registered 50-step micro gate passes. This is a `TRAINING_PASS` and `VIDEOPAINTER_ADAPTER_POSITIVE` for the locked search-dev micro experiment only. It is not a final benchmark result and does not start longer training.
