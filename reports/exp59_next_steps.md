# Exp59 Next Steps

Status: `VOID_TARGET_HIT_WEAK_REGENERATE_DATA`

## Next Minimal Experiment

Run a small Kubric regeneration experiment before any training:

- Generate exactly Gate8 again.
- Require `target_hit=true` rows.
- Prefer a larger or official-compatible render resolution if runtime allows.
- Preserve `rgb_full.mp4`, `rgb_removed.mp4`, `mask.mp4` or `quadmask_0.mp4`, and `metadata.json`.
- Review all 8 native evidence pages.
- Run official VOID pass1 inference again.
- Compute the same full and quadmask-aware metrics.
- Only if native data is target-hit-positive and produces useful medium-hard same-model losers should a Kubric one-step diagnostic be considered.

## Do Not Do Yet

- Do not run one-step on the current Exp58B Gate8.
- Do not run 10-step.
- Do not expand to Gate16/Gate32 before Gate8 target-hit-positive review.
- Do not change VOID loss or trainable scope before native data quality is repaired.

## Rationale

Exp59 shows the runtime/inference path works. The blocker is data quality: all rows are `target_hit=false`, and transition residuals remain common. Training on this would confound data mismatch with weak target generation.
