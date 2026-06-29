# Exp42 Metric Summary

No Exp42 metrics have been generated yet.

Readback imports previous MiniMax evidence:

- Exp36 perturbation full/mask MAE: `0.088218` / `0.156302`.
- Exp37 R1 heldout full/mask/boundary/outside deltas: `+0.200826` /
  `+0.161946` / `-0.049755` / `+0.028198`; visual better `1/16`.
- Exp38 R1 heldout13 full/mask/boundary/outside deltas: `+0.102167` /
  `+0.117230` / `-0.141510` / `-0.037262`; visual pass false.
- Exp40 best SFT grid full/mask/boundary/outside deltas: `-1.816781` /
  `-1.634597` / `-1.899575` / `-2.624405`.

## 2026-06-29 Official Successful-Removal Mining

Status: `MINIMAX_SUCCESSFUL_REMOVAL_POOL_WEAK`.

Automatic mining metrics:

- sources: `117`
- seeds/source: `4`
- candidates: `468`
- technical-valid: `468`
- successful candidates: `52`
- medium-hard failure candidates: `80`
- failed candidates: `0`
- full PSNR mean: `24.870348`
- mask PSNR mean: `19.644484`
- boundary PSNR mean: `20.910670`
- outside PSNR mean: `27.539608`
- temporal-diff MAE mean: `2.568904`

Automatic classification counts:

- `SUCCESSFUL_REMOVAL_CANDIDATE`: `52`
- `MEDIUM_HARD_REMOVAL`: `80`
- `OUTSIDE_BAD`: `112`
- `FOGGING_OVER_ERASURE`: `100`
- `TRIVIAL_BAD`: `56`
- `BOUNDARY_BAD`: `37`
- `TOO_CLOSE`: `31`

Codex post-review source-level metrics:

- success rows: `52`
- success scene groups: `18`
- failure rows: `80`
- failure scene groups: `29`
- success/failure scene overlap: `7`
- required overlap for bad-noise v3: `24`
- auto-failures marked visually noisy/borderline at strip scale: `37/80`

Decision: row-level yield is real, but source-level paired signal is
insufficient. Bad-noise v3, Stage2 train/search/shadow, SFT, and DPO remain
locked.
