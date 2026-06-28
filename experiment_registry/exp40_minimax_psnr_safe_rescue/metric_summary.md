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

## 2026-06-28 R1 Sample-Level Diagnosis

Available existing evidence:

- Exp38 SFT/DPO R1 heldout13: full/mask/boundary/outside means
  `+0.102167` / `+0.117230` / `-0.141510` / `-0.037262`.
- Exp38 train-overfit Exp37 R1 train32: full/mask/boundary/outside means
  `-0.586255` / `+0.152062` / `+0.069123` / `-0.895018`.
- Exp38 train-overfit Exp37 R1 heldout16: full/mask/boundary/outside means
  `+0.200826` / `+0.161946` / `-0.049755` / `+0.028198`.

The train32 evidence shows mask/boundary motion but full/outside regression;
the heldout evidence shows the desired full/mask direction but boundary is not
safe. This points to PSNR-safe SFT with stronger outside and boundary
preservation before any DPO-after-SFT gate.
