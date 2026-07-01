# Exp53 H20 VOID R1/R2 Targeted One-Step Rescue

Status: `EXP53B_ONESTEP_MIXED_ONLY`

H20 lane only: R1/R2 on GPU0-3.

Boundaries: one-step only, no VOR-Eval, no hard comp, no 10-step, no third-backbone claim.

## Milestone A - 2026-07-01T15:59:54+08:00

Status: `EXP53_GPU_BLOCKED`. Readback and GPU audit completed. H20 lane only: R1/R2 on GPU0-3.

## Milestone B - 2026-07-01T08:53:57+00:00

Status: `EXP53_R1R2_PREREGISTERED`. Wave1 cells locked for R1/R2 on H20 GPU0-3; 10-step remains locked for Exp55 aggregation. Current GPU status: `EXP53_H20_PARTIAL_GPU_READY`.

## Milestone C - 2026-07-01T09:07:39+00:00

Status: `EXP53_R1R2_ONESTEP_BLOCKED`. R2_Q2_T500_S0 was attempted on GPU2 but produced no checkpoint after 10m03s; GPU0/GPU3 were occupied by unrelated processes and T300 cache was unavailable. No 10-step.

## Milestone D - 2026-07-01T09:09:15+00:00

Status: `EXP53_WAVE2_NOT_RUN`. No Wave1 PASS/MIXED-safe cell existed; Wave2 and 10-step remain locked.

## Milestone E - 2026-07-01T09:10:46+00:00

Status: `EXP53_H20_BLOCKED`. Final handoff complete. No PASS/MIXED candidate and no 10-step.

## Exp53B Milestone A - 2026-07-01T15:20:19+00:00

Status: `EXP53B_READY_FOR_CORE_CELLS`. GPU/cache audit complete for R1_Q2_T500_S0 and R2_Q2_T500_S0 only.

## Exp53B Milestone C - 2026-07-02T00:52:00+08:00

Status: `EXP53B_CORE_ONESTEP_MIXED`. The core recovery produced checkpoints, strict reloads, and heldout4 Step0/Step1 videos for both `R1_Q2_T500_S0` and `R2_Q2_T500_S0`.

Mean heldout deltas:

| Cell | Full PSNR | Object PSNR | Overlap PSNR | Affected PSNR | Boundary PSNR | Outside PSNR | Visual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R1_Q2_T500_S0 | +0.020812 | +0.803855 | -0.127050 | -0.069499 | -0.052405 | +0.049764 | 0 better / 2 tie / 2 worse |
| R2_Q2_T500_S0 | -0.007600 | +0.933518 | -0.223775 | -0.171283 | -0.065547 | +0.053105 | 0 better / 2 tie / 2 worse |

Decision: no one-step PASS; do not run 10-step locally. R1_Q2_T500_S0 is the best H20 candidate but remains mixed because overlap/affected/boundary regressions persist.

## Exp53B Milestone D - 2026-07-02T01:02:00+08:00

Status: `EXP53B_ONESTEP_MIXED_ONLY`. Final handoff complete. `R1_Q2_T500_S0` is ranked first, `R2_Q2_T500_S0` second, and neither unlocks local 10-step. Exp55 should aggregate with Exp54 before any further micro-gate decision.
