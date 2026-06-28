# Exp31 Metric Summary

Final status: `VIDEOPAINTER_2000_POSITIVE`

Exp31 L0/L1 technical metrics:

- run id: `exp31_vp_l0_l1_20260627_132158`
- L0 loss: `0.695064902305603`
- L0 DPO loss: `0.6931471824645996`
- policy grad norm: `14.379269412062548`
- reference has grad: `false`
- L1 policy delta norm: `1.6732703166152714`
- L1 reference delta norm: `0.0`
- strict reload max abs diff: `0.0`

Quality metrics:

| split | comparison | win rate | full PSNR delta | mask PSNR delta | sampled boundary PSNR delta | sampled outside L1 delta | temporal delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| search | Step2000 vs Step0 | 0.9688 | +5.5701 | +9.9747 | +12.0920 | +0.7533 | -0.5468 |
| search | Step2000 vs Step50 | 1.0000 | +6.1338 | +1.8747 | +3.7226 | -10.0351 | +0.2022 |
| shadow | Step2000 vs Step0 | 1.0000 | +6.2632 | +10.8860 | +12.2343 | +0.7666 | -0.5314 |
| shadow | Step2000 vs Step50 | 1.0000 | +6.4772 | +2.0832 | +3.9405 | -10.5232 | +0.2140 |

LPIPS and Ewarp completion is recorded in
`reports/exp31_vp_2000_lpips_ewarp_metrics.md`.

Formal shadow-dev comp deltas:

| comparison | full PSNR | full LPIPS | mask PSNR | mask LPIPS | boundary PSNR | boundary LPIPS | Ewarp mask | win/probability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Step2000 vs Step0 | +11.440561 | -0.056840 | +11.440561 | -0.213718 | +15.242894 | -0.207700 | -11.171650 | 1.0000 |
| Step2000 vs Step50 | +2.305730 | -0.008813 | +2.305730 | -0.034082 | +3.637059 | -0.033266 | -0.258536 | >=0.9062 |

The formal VideoPainter-only gate is satisfied: Step2000 improves over Step0,
is not worse than Step50, LPIPS improves rather than regresses, mask-region
Ewarp improves rather than regresses, and the completed video review found no
systemic new artifact. This is not a universal-adapter, final-SOTA,
all-models-supported, or top-conference novelty claim.

Strict validation readback is complete in
`reports/exp31_vp_2000_strict_readback.md`. Official base identity replay
passed in `reports/exp31_vp_2000_base_identity_audit.md`: official base and
Step0 weights match, replay-vs-existing raw/comp frames are exact on 2
search-dev + 2 shadow-dev rows for Step0/50/2000, and comp formula/polarity
recomputation is exact.

Metric caveats:

- TC is recorded as `TC_BACKEND_NOT_LOCAL`; no automatic TC model download was
  triggered and no proxy is reported as real TC.
- Ewarp in the formal completion is mask-region Ewarp from the existing
  `inference/metrics.py` backend with OpenCV DIS fallback because RAFT weights
  were not local on PAI.
- Comp outside pixels are copied from the winner by protocol, so outside L1 is
  exactly `0.0` and is not model-predicted outside preservation evidence.
