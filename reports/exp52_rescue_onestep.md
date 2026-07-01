# Exp52 VOID Rescue One-Step Grid

Status: `VOID_RESCUE_ONESTEP_MIXED`

Wave 1 ran the preregistered all-GPU one-step forward/checkpoint grid where GPUs were available. Seven cells completed a one-step checkpoint; `R4_Q2_T500_S0` was skipped because GPU7 had an unrelated external process and was not killed.

Full heldout video evidence was generated for `R1_Q0_T500_S0`, because it was the clearest forward-ready cell and has a valid Step0 baseline. Codex opened the contact sheet and inspected all four heldout samples.

## R1_Q0_T500_S0 Mean Step1 - Step0

- full PSNR: 0.01562676520194195
- object PSNR: 1.025830221887892
- overlap PSNR: -0.11671521679298635
- affected PSNR: -0.11865014078788594
- affected-union PSNR: -0.11432922107159271
- boundary PSNR: 0.1608491675666972
- outside PSNR: 0.04482370447721884
- SSIM: -0.00011039303058779648
- Step0-Step1 L1: 0.009048108409236495

## Interpretation

R1 reduces the Exp50 loser-dominant failure mode in the intended direction: object, boundary, outside, and full PSNR are safe or positive, with no collapse and no systematic outside damage. However, affected and overlap regions regress, and the visual review is tie-heavy rather than positive. This is not enough to unlock 10-step training under Exp52 gates.

Visual review: {'better': 0, 'tie': 3, 'worse': 1}

No VOR-Eval, hard comp, or long training was used.
