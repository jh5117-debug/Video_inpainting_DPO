# Exp50 VOID Next Steps V2

## Recommendation

Stop the current VOID 10-step adapter direction as a positive-evidence path. Keep VOID as a baseline and loser generator. Continue third-model search or revisit VOID only with a redesigned objective/trainable subset.

## Why

- Inference smoke passed on VOR Gate8: VOID is usable on PAI/H20.
- Preference forward, zero-gap, and one-step are technically valid.
- The 10-step heldout micro gate did not improve enough local/effect metrics and did not show clear visual improvement.

## Minimal Future Options

1. Use VOID outputs as same-model medium-hard loser candidates for a separate preference dataset.
2. Try a more localized objective or a broader trainable subset only if there is a clear hypothesis and a new micro gate.
3. Resume ROSE or another third-model candidate rather than scaling VOID training now.

## Explicit Non-Recommendations

- Do not run 30/50/100/300/500-step VOID training from this checkpoint.
- Do not promote VOID as third adapter evidence.
- Do not claim universal adapter or final SOTA.
