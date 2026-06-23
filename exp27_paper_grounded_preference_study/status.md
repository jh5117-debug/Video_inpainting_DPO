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
