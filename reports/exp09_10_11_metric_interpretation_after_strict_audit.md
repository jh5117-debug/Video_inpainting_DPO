# Exp9/10/11 Metric Interpretation After Strict-Mask Audit

Date: 2026-06-11

## Current Fixed DAVIS50 Table

The existing table is a DAVIS50 frame-wise raw6 hard-comp table with LPIPS,
VFID, and TC from the same all-metric pass. It is valid as whole-frame
hard-comp evidence.

It is **not** strict mask-pixel evidence because strict mask-pixel PSNR was not
present in the previous table.

## Answers

### Does Exp10-1 remain best on whole-frame PSNR/SSIM?

No. In the current whole-frame table:

- best PSNR: Exp10-2 (`32.9491`)
- best SSIM: Exp10-2 (`0.9723`)
- best LPIPS / TC: Exp10-1 is best or close to best

### Does Exp10-1 also win strict mask-pixel PSNR?

Unknown until the patched wrapper is rerun. The previous table does not contain
`strict_mask_pixel_psnr`.

### Is Exp11 a real method result?

No. Exp11 is invalid / mislabeled / blocked as a flow-prior consistency method.
Existing Exp11 rows should be treated as historical proxy rows and should not
be used as method claims.

### Can current results be called SOTA?

No. The current defensible statement is narrower:

```text
Under our fixed DAVIS-50 frame-wise raw6 hard-comp protocol, Exp10 variants
improve whole-frame PSNR/SSIM over the SFT48000 DiffuEraser baseline.
Strict mask-pixel PSNR must be rerun with the patched wrapper before claiming
mask-region superiority. Exp11 rows are invalid as flow-prior consistency
results.
```

## Required Table Correction

Keep Exp11 numeric rows only if explicitly labeled:

```text
INVALID / MISLABELED proxy, not true flow-prior consistency DPO.
```
