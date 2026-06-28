# Exp40 Metric Summary

No Exp40 model outputs have been generated yet.

Readback imported Exp38 R1 heldout13 metrics:

- full PSNR delta: `+0.102167`
- mask PSNR delta: `+0.117230`
- boundary PSNR delta: `-0.141510`
- outside PSNR delta: `-0.037262`
- full-positive rows: `7/13`
- mask-positive rows: `7/13`
- boundary-negative rows: `9/13`
- outside-negative rows: `8/13`
- outside-MAE-worse rows: `10/13`

This is not a positive gate. Exp40 must exceed `+0.2 dB` on shadow while making
boundary/outside safe and preserving LPIPS/Ewarp.
