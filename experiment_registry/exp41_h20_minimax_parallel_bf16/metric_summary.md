# Exp41 Metric Summary

No Exp41 MiniMax training or evaluation metric has been run yet.

Imported baselines:

- Exp38 R1 heldout13 full/mask/boundary/outside PSNR deltas:
  `+0.102167` / `+0.117230` / `-0.141510` / `-0.037262`.
- Exp40 LocalDPO v3 minimum pool:
  `train64/search24/shadow24`.
- Exp40 Step0 raw baseline:
  - train full/mask/boundary/outside PSNR:
    `23.965598` / `18.485359` / `19.395954` / `26.458319`
  - search full/mask/boundary/outside PSNR:
    `25.043807` / `20.493872` / `21.409812` / `27.765446`
  - shadow full/mask/boundary/outside PSNR:
    `26.209732` / `21.645338` / `24.277694` / `29.577002`

Exp41 success requires shadow raw full PSNR at least `+0.20 dB` over Step0
with mask/boundary/outside, LPIPS, Ewarp, and visual safety.

## 2026-06-29 Data Audit

No Exp41 training metric has been run. Data readiness passed: `2242` active
manifest refs validated on H20 with `0` missing, and Exp40 decode audit had
`0` failures. This does not change MiniMax quality status.

## 2026-06-29 BF16 Preflight

Runtime-only preflight passed. P0-P7 were finite, including P7 DDP8 bf16-safe
one-batch training. No model-quality metric was run in this milestone.
