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

## 2026-06-29 Official Protocol Audit

Protocol-smoke metrics were computed on `4` train and `4` search rows. These
metrics validate the executable protocol path only; they are not a promotion
evaluation.

| label | split | full PSNR | mask PSNR | boundary PSNR | outside PSNR | outside MAE | temporal diff MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| official_readme_test | train4 | 28.014237 | 20.634074 | 21.408269 | 30.824739 | 4.555740 | 1.710896 |
| official_readme_test | search4 | 27.342555 | 20.120630 | 21.166384 | 33.812574 | 3.230408 | 1.183885 |
| feature_6step_probe | train4 | 28.027605 | 20.710534 | 21.539239 | 30.863828 | 4.466146 | 1.729045 |
| feature_6step_probe | search4 | 27.547341 | 20.459429 | 20.962746 | 33.876004 | 3.137951 | 1.259875 |

The 6-step probe is diagnostic only because the executable README/test protocol
uses 12 steps. It does not replace the current Exp40/H20 Step0 protocol.
