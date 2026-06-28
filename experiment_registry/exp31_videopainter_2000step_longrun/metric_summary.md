# Exp31 Metric Summary

Final status: `VIDEOPAINTER_2000_PARETO_MIXED`

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

LPIPS and Ewarp were not computed in this fast summary, so the formal
`VIDEOPAINTER_2000_POSITIVE` gate is blocked even though the available metrics
favor Step2000.

Strict validation readback is complete in
`reports/exp31_vp_2000_strict_readback.md`. LPIPS/Ewarp and replay identity
remain pending before formal promotion.
