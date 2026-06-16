# Exp15 OR Metric Protocol

This benchmark is **object removal (OR)**, not the previous BR / inpainting
hard-comp protocol.

## Dataset

- Source masks: DAVIS2017 foreground annotations.
- Eval split: DAVIS50 only.
- Mask semantics: `mask != 0` means foreground object to remove.

## Metric Rule

No compositing is used before scoring.

```text
input  = original DAVIS frame
output = raw method prediction
mask   = DAVIS foreground annotation
bg     = mask == 0
```

Primary metrics:

- `PSNR_bg`: strict pixel PSNR on background pixels only.
- `SSIM_bg_ignore_mask`: foreground pixels are ignored by setting foreground
  to black in both input and output before SSIM. This is a background
  preservation proxy, not arbitrary-pixel strict SSIM.
- `TC_bg`: simple background temporal-difference consistency if outputs are
  available.

Optional metrics:

- `LPIPS` / `VFID` are only reported if the existing project wrappers support
  them cleanly for OR. They are not fabricated.

## Interpretation

OR has no ground-truth "removed object" background inside the mask. Therefore:

- mask-inside PSNR / SSIM is not a main metric;
- visual quality inside the removed object region must be judged qualitatively;
- OR tables must stay separate from BR / hard-comp full-frame tables.

