# Exp59 Kubric Official Inference Metrics And Visual Review
Status: `VOID_KUBRIC_INFERENCE_REVIEW_WEAK`
Official VOID pass1 outputs are technically valid on all 8 Kubric-native Gate8 rows, but all 8 rows retain the Exp58B `target_hit=false` caveat. Visual review found stable outside/background behavior, while object/overlap/affected/boundary residuals remain too common for adapter-data promotion.
## Metric Protocol
- Compared against `rgb_removed` in the common native diagnostic space.
- Frame window: first 24 frames.
- Official output was downscaled from 384x672 to 128x128 for native metrics.
- LPIPS/Ewarp/TC are `NA` in this no-training diagnostic path.

## Mean Metrics
- full_psnr: `30.152555`
- ssim: `0.919492`
- object_psnr: `28.337691`
- overlap_psnr: `16.673219`
- affected_psnr: `17.527094`
- boundary_psnr: `22.267098`
- outside_psnr: `34.210532`
- outside_l1: `0.014922`
- temporal_flicker: `0.012497`
- object_residual: `0.025721`
- effect_residual: `0.078867`
- tone_drift: `0.009077`
- output_input_l1: `0.017905`
- output_gt_l1: `0.018055`

## Visual Review Summary
- Evidence packs opened: 8/8 contact sheets.
- Output technically valid: 8/8.
- Outside/background stable or safe: 8/8.
- `target_hit=false`: 8/8.
- Medium-hard loser diagnostics: 2/8.
- Too-close/weak diagnostics: 2/8.
- Transition damage / residuals: 6/8.

## Per-Sample Review
- `00000`: `KUBRIC_TRANSITION_DAMAGE;KUBRIC_TARGET_HIT_WEAK`. Output is technically valid and outside/background is mostly stable, but the target object remains visible and overlap/affected regions do not match the removed-object target.
- `00001`: `KUBRIC_TRANSITION_DAMAGE;KUBRIC_TARGET_HIT_WEAK`. White object residual remains visible through affected/boundary crops; outside is preserved but removal target is weak.
- `00002`: `KUBRIC_MEDIUM_HARD_LOSER;KUBRIC_TARGET_HIT_WEAK;KUBRIC_TRANSITION_DAMAGE`. Plausible same-model loser candidate with stable outside and some object-region change, but residual foreground and transition mismatch remain.
- `00003`: `KUBRIC_TRANSITION_DAMAGE;KUBRIC_TARGET_HIT_WEAK`. Foreground target persists across object/overlap/boundary sheets; background does not collapse but output is not a clean removal.
- `00004`: `KUBRIC_TOO_CLOSE;KUBRIC_TARGET_HIT_WEAK`. Metrics are high and background is stable, but target is very small and residual remains; this is too-close/weak as training evidence.
- `00005`: `KUBRIC_TRANSITION_DAMAGE;KUBRIC_TARGET_HIT_WEAK`. Target and interaction-region residuals are strong; output is technically valid but weak as a medium-hard loser.
- `00006`: `KUBRIC_TOO_CLOSE;KUBRIC_TARGET_HIT_WEAK`. High full/outside metrics and stable background, but object remains and target-hit=false makes this too-close/weak diagnostic data.
- `00007`: `KUBRIC_MEDIUM_HARD_LOSER;KUBRIC_TARGET_HIT_WEAK;KUBRIC_TRANSITION_DAMAGE`. Potential medium-hard loser with stable outside, but visible residual in affected/boundary regions keeps the sample weak for adapter data.

## Decision
This is a valid official-inference diagnostic and same-model loser-generation smoke on Kubric-native data. It is not adapter evidence and not ready for one-step training because `target_hit=false` affects all rows and transition-region residuals remain common.
