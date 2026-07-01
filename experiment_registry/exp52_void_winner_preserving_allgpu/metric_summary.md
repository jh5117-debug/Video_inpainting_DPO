# Exp52 Metric Summary

Milestone B profile:

- component load: 55.16 sec
- transformer load: 263.47 sec
- cache rows: 32
- parity: `VOID_CACHE_PARITY_EXPLAINED`

Milestone C R1 row0:

- status: `VOID_R1_ROW0_SMOKE_PASS`
- runtime: 632.97 sec
- max param delta norm: 0.005100787617266178
- winner_gap_post: 0.0001377984881401062
- loser_gap_post: 1.9371509552001953e-06
- loser contribution ratio: 0.014258294488620784
- grad finite: True
- reload ok: True
- peak reserved VRAM: 20.053 GiB

## Milestone E Wave1 One-Step Grid - 2026-07-01T14:53:24+08:00

Status: `VOID_RESCUE_ONESTEP_MIXED`

- forward checkpoints: 7/8 preregistered cells
- skipped: `R4_Q2_T500_S0` because GPU7 had an unrelated external process
- video-evaluated cell: `R1_Q0_T500_S0`
- full PSNR delta: 0.01562676520194195
- object PSNR delta: 1.025830221887892
- overlap PSNR delta: -0.11671521679298635
- affected PSNR delta: -0.11865014078788594
- boundary PSNR delta: 0.1608491675666972
- outside PSNR delta: 0.04482370447721884
- SSIM delta: -0.00011039303058779648
- visual: 0 better / 3 tie / 1 worse
- decision: 10-step not unlocked because affected/overlap regressed and visual evidence is not positive.

## Milestone F Wave2 Scope Decision - 2026-07-01T14:55:45+08:00

Status: `EXP52_WAVE2_DEFERRED_NO_ONESTEP_PASS`. No additional metrics were generated; 10-step remains locked.
