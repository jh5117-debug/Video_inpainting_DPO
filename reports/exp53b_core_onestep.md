# Exp53B Core One-Step Cells

Status: `EXP53B_CORE_ONESTEP_MIXED`

Scope: only `R1_Q2_T500_S0` and `R2_Q2_T500_S0`; one optimizer step; no 10-step.

## Cell Summary
### R1_Q2_T500_S0
- status: `MIXED`
- checkpoint: `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp53_void_r1r2_targeted_h20/core_forward/R1_Q2_T500_S0/checkpoints/R1_Q2_T500_S0_adapter_step1.pt`
- contact sheet: `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp53_void_r1r2_targeted_h20/core_video/R1_Q2_T500_S0/R1_Q2_T500_S0_heldout_contact_sheet.jpg`
- full PSNR delta: 0.020812
- object PSNR delta: 0.803855
- overlap PSNR delta: -0.127050
- affected PSNR delta: -0.069499
- boundary PSNR delta: -0.052405
- outside PSNR delta: 0.049764
- SSIM delta: -0.000100
- heldout loser contribution ratio: 0.561391
- visual counts: {'better': 0, 'tie': 2, 'worse': 2}

### R2_Q2_T500_S0
- status: `MIXED`
- checkpoint: `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp53_void_r1r2_targeted_h20/core_forward/R2_Q2_T500_S0/checkpoints/R2_Q2_T500_S0_adapter_step1.pt`
- contact sheet: `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp53_void_r1r2_targeted_h20/core_video/R2_Q2_T500_S0/R2_Q2_T500_S0_heldout_contact_sheet.jpg`
- full PSNR delta: -0.007600
- object PSNR delta: 0.933518
- overlap PSNR delta: -0.223775
- affected PSNR delta: -0.171283
- boundary PSNR delta: -0.065547
- outside PSNR delta: 0.053105
- SSIM delta: -0.000106
- heldout loser contribution ratio: 0.603183
- visual counts: {'better': 0, 'tie': 2, 'worse': 2}

## Exp52 R1_Q0 Baseline Reference
{
  "affected_psnr_delta": -0.11865014078788594,
  "boundary_psnr_delta": 0.1608491675666972,
  "full_psnr_delta": 0.01562676520194195,
  "object_psnr_delta": 1.025830221887892,
  "outside_psnr_delta": 0.04482370447721884,
  "overlap_psnr_delta": -0.11671521679298635,
  "ssim_delta": -0.00011039303058779648
}

No VOR-Eval, hard comp, 10-step, or long training was used.
