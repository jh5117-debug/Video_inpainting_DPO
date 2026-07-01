# Exp52 VOID All-GPU Rescue Preregistration

Status: `EXP52_RESCUE_GRID_PREREGISTERED`

## Preconditions

- Cache gate: `VOID_CACHE_PARITY_EXPLAINED`
- Row0 smoke: `VOID_R1_ROW0_SMOKE_PASS`
- No VOR-Eval
- No hard comp
- No direct long training
- No 10-step before one-step PASS

## Wave 1

| GPU | Cell | Recipe | Quadmask | Timestep | Scope |
| --- | --- | --- | --- | --- | --- |
| 0 | R1_Q0_T500_S0 | R1_WinnerPreserve_LocalDPO | Q0_current | 500 | S0_proj_out |
| 1 | R1_Q2_T500_S0 | R1_WinnerPreserve_LocalDPO | Q2_strict_affected | 500 | S0_proj_out |
| 2 | R2_Q0_T500_S0 | R2_WinnerPreserve_LoserClip | Q0_current | 500 | S0_proj_out |
| 3 | R2_Q2_T500_S0 | R2_WinnerPreserve_LoserClip | Q2_strict_affected | 500 | S0_proj_out |
| 4 | R3_Q0_T500_S0 | R3_SDPO_Safe | Q0_current | 500 | S0_proj_out |
| 5 | R3_Q2_T500_S0 | R3_SDPO_Safe | Q2_strict_affected | 500 | S0_proj_out |
| 6 | R4_Q0_T500_S0 | R4_LinearDPO_vPrediction | Q0_current | 500 | S0_proj_out |
| 7 | R4_Q2_T500_S0 | R4_LinearDPO_vPrediction | Q2_strict_affected | 500 | S0_proj_out |

GPU7 is preregistered but must be rechecked before launch because D readback saw an unrelated external process there.

## Wave 2

Wave 2 is conditional. It runs only if Wave 1 has zero clear PASS but at least mixed signals, or one PASS needing nearby ablation. If Wave 1 gives at least two PASS recipes, Wave 2 is skipped and top recipes go to 10-step.

| GPU | Cell | Recipe | Quadmask | Timestep | Scope |
| --- | --- | --- | --- | --- | --- |
| 0 | R1_Q1_T500_S0 | R1_WinnerPreserve_LocalDPO | Q1_object_only | 500 | S0_proj_out |
| 1 | R2_Q1_T500_S0 | R2_WinnerPreserve_LoserClip | Q1_object_only | 500 | S0_proj_out |
| 2 | R3_Q1_T500_S0 | R3_SDPO_Safe | Q1_object_only | 500 | S0_proj_out |
| 3 | R4_Q1_T500_S0 | R4_LinearDPO_vPrediction | Q1_object_only | 500 | S0_proj_out |
| 4 | R1_Q2_T300_S0 | R1_WinnerPreserve_LocalDPO | Q2_strict_affected | 300 | S0_proj_out |
| 5 | R2_Q2_T300_S0 | R2_WinnerPreserve_LoserClip | Q2_strict_affected | 300 | S0_proj_out |
| 6 | R3_Q2_T300_S0 | R3_SDPO_Safe | Q2_strict_affected | 300 | S0_proj_out |
| 7 | R4_Q2_T300_S0 | R4_LinearDPO_vPrediction | Q2_strict_affected | 300 | S0_proj_out |

## Recipes

- R1 WinnerPreserve-LocalDPO: DPO only object+affected, winner anchor 0.05, outside preservation 0.10, boundary preservation 0.05, loser_grad_scale 0.0.
- R2 WinnerPreserve-LoserClip: object+affected+boundary, winner anchor 0.05, outside 0.10, boundary 0.05, loser_gap_clip_tau 0.0005, loser_grad_scale 0.1.
- R3 SDPO-Safe: safe loser scale clipped to [0.0, 0.25], winner anchor 0.05, outside 0.10, boundary 0.05.
- R4 LinearDPO-vPrediction: linear v-prediction utility, frozen reference, winner anchor 0.05, loser clipping enabled, local region only.

## Gates

One-step PASS requires full/outside safety, local region non-regression, visual better/tie >= 3/4, no collapse/tone drift/boundary destruction, loser contribution <= 50%, and winner gap non-negative or winner absolute loss not worse.

10-step remains locked until at least one one-step recipe passes.
