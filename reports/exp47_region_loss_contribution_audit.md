# Exp47 Region Loss Contribution Audit

Status: `EXP47_REGION_LOSS_GLOBAL_DRIFT_RISK_CONFIRMED`

Rows audited: `16` (`8` train-batch proxy rows, `8` search-batch proxy rows). This is a no-training, no-optimizer frame-space proxy audit of Exp46 SFT-B region weights and pseudo-success target participation.

## Runner Formula Readback

Exp43/Exp46 SFT runner builds weights as:

```text
weight = far_outside global base
       + (outside - far_outside) * outside
       + mask * mask_weight
       + boundary * boundary_weight
       + affected * normalized_abs(condition - winner)
```

Configured SFT-B weights: mask `0.75`, boundary `1.5`, affected `0.75`, outside `0.2`, far_outside `0.03`.

Important implementation detail: the computed far-outside region is not used as a region-specific component in `build_region_weight`; `far_outside` is a global base weight. Also, `affected` is computed from condition versus pseudo-success winner and normalized, so pseudo-success global drift can add weight outside the local mask.

## Mean Areas And Contributions

- mask/boundary/outside area: `0.019470` / `0.022919` / `0.968317`
- affected mean / affected outside mean: `0.028255` / `0.020099`
- weight mean / max: `0.264786` / `3.030000`
- normalized component contribution far-base/outside/mask/boundary/affected: `0.114598` / `0.630115` / `0.052052` / `0.123524` / `0.079712`
- outside + affected + global-base component contribution: `0.824424`
- condition-vs-pseudo outside L1: `0.016474`
- pseudo-vs-V_bg outside L1: `0.016277`

## Conclusion

The region weighting implementation is finite and mask polarity is sane, but it is not safely localized for drifting pseudo-success targets. Outside receives nonzero loss, far-outside is a global base, and the affected term can spread pseudo target differences outside the object region. This confirms a global-drift risk in the SFT objective even without an obvious manifest bug.
