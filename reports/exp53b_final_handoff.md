# Exp53B Final Handoff

Status: `EXP53B_ONESTEP_MIXED_ONLY`

Exp53B recovered the two core H20 cells requested:

- `R1_Q2_T500_S0`
- `R2_Q2_T500_S0`

Both cells produced one-step checkpoints, strict reloads, and heldout4 Step0/Step1 video evidence. No 10-step was run.

## Answers

1. Did `R1_Q2_T500_S0` produce checkpoint? Yes. Checkpoint: `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp53_void_r1r2_targeted_h20/core_forward/R1_Q2_T500_S0/checkpoints/R1_Q2_T500_S0_adapter_step1.pt`.
2. Did `R2_Q2_T500_S0` produce checkpoint? Yes. Checkpoint: `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp53_void_r1r2_targeted_h20/core_forward/R2_Q2_T500_S0/checkpoints/R2_Q2_T500_S0_adapter_step1.pt`.
3. Did R1 reduce loser dominance? On train it removes loser push by construction; heldout diagnostic loser contribution remains 0.561391, but R1 is still safer than R2 in local metrics.
4. Did R2 loser clipping help? No clear promotion. R2 heldout loser contribution is 0.603183 and local overlap/affected/boundary regressions are stronger than R1.
5. Did Q2 strict affected reduce affected/overlap regression compared with Exp52 R1_Q0? Mixed. R1_Q2 improves affected regression versus R1_Q0 (-0.069499 vs -0.118650), but overlap is slightly worse (-0.127050 vs -0.116715) and boundary flips from positive to slightly negative (-0.052405 vs +0.160849).
6. Which cell is best? `R1_Q2_T500_S0`.
7. Is any cell one-step PASS? No. Both are mixed-only.
8. Should Exp55 aggregator run 10-step or not? Not from Exp53B alone. Exp55 should aggregate Exp53B and Exp54, but no local 10-step is unlocked by this lane.

## Metrics

| Cell | Status | Full PSNR | Object PSNR | Overlap PSNR | Affected PSNR | Boundary PSNR | Outside PSNR | Visual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R1_Q2_T500_S0 | MIXED | +0.020812 | +0.803855 | -0.127050 | -0.069499 | -0.052405 | +0.049764 | 0 better / 2 tie / 2 worse |
| R2_Q2_T500_S0 | MIXED | -0.007600 | +0.933518 | -0.223775 | -0.171283 | -0.065547 | +0.053105 | 0 better / 2 tie / 2 worse |

No VOR-Eval, hard comp, 10-step, long training, universal adapter, final SOTA, or third-backbone claim was used.
