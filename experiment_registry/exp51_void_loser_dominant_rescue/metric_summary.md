# Exp51 Metric Summary

Milestone A status: `VOID_LOSER_DOMINANT_CONFIRMED`. Mean Exp50 10-step deltas: full -0.000965, mask -0.229878, affected 0.019341, boundary -0.063034, outside 0.043422.

## Milestone B Quadmask Metrics

- Status: `VOID_QUADMASK_METRICS_READY`
- 10-step affected_union delta PSNR: -0.241561
- 10-step overlap delta PSNR: -0.423390
- 10-step object_core delta PSNR: -0.569839
- 10-step outside_background delta PSNR: 0.029117

## Milestone C - SFT Parity

- Target parameterization: `v_prediction`
- SFT loss: official mean MSE mirrored by wrapper
- Strict helper parity: blocked by official Accelerator loop encapsulation
- Status: `VOID_SFT_PARITY_EXPLAINED_ONLY`

## Milestone D - Quadmask Ablation Data

- Variants: Q0 current, Q1 object-only, Q2 strict affected, Q3 broad affected
- Rows: 8 per variant
- Visual sheets opened: 8/8
- Preferred diagnostic priority: Q1/Q2 before Q3

## Milestone E - Native Kubric

No metrics: native data generation blocked before videos exist. Blockers: Kubric/PyBullet/Blender/HUMOTO assets.

## Milestone F - Preregistration

No metrics by design. Gates fixed: one-step first, 10-step only if one-step passes.
