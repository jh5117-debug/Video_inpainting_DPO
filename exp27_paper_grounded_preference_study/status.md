# Exp27 Status

- PAPER_REVIEW_COMPLETE
- EXACT_BASELINE_REPRODUCTION_IN_PROGRESS
- NO_LONG_TRAINING
- LOCALDPO_COMPAT_MASK_ONLY_PASSED
- SDPO_LINEAR_REAL_BATCH_PARITY_PENDING

## Gate Notes

- LocalDPO official random-mask raw path is blocked by a matplotlib ARGB/RGB compatibility issue in the official cached code path.
- Exp27 isolated compatibility wrapper passes mask-only deterministic probes without editing the official clone.
- Diffusion-SDPO scalar safe-lambda toy parity passed exactly.
- Linear-DPO utility and EMA toy parity passed exactly.
- No long training has been launched from Exp27.

## 2026-06-23 LocalDPO Fusion Primitive

- Added isolated LocalDPO latent fusion / progressive outside reinjection helper.
- Unit tests verify outside-latent preservation and separate task/corruption/restoration masks.
- CPU parity script now writes `localdpo_full_parity.json`.
- Real DiffuEraser-batch SDPO and Linear-DPO parity remain pending; no studies or RC-FPO runs started.

Status: `LOCALDPO_FUSION_PRIMITIVE_READY_REAL_BATCH_PENDING`.
