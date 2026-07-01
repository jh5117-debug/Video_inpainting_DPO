# Exp53 Metric Summary

Status: `EXP53_H20_BLOCKED`. No Step1 metric deltas were produced because no checkpoint/video evidence was generated.


## Exp53B Readback

Status: `EXP53B_READY_FOR_CORE_CELLS`. No metrics yet; Q2/T500 cache is ready: `True`.

## Exp53B Core One-Step

Status: `EXP53B_CORE_ONESTEP_MIXED`.

| Cell | Full PSNR | Object PSNR | Overlap PSNR | Affected PSNR | Boundary PSNR | Outside PSNR | SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R1_Q2_T500_S0 | +0.020812 | +0.803855 | -0.127050 | -0.069499 | -0.052405 | +0.049764 | -0.000100 |
| R2_Q2_T500_S0 | -0.007600 | +0.933518 | -0.223775 | -0.171283 | -0.065547 | +0.053105 | -0.000106 |

R1_Q2_T500_S0 improved full/object/outside but missed the one-step PASS gate because overlap, affected, and boundary regions regressed. R2_Q2_T500_S0 worsened local spill more strongly. LPIPS/Ewarp were not available in this isolated audit.
