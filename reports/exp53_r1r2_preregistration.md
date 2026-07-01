# Exp53 H20 R1/R2 Preregistration

Status: `EXP53_R1R2_PREREGISTERED`

Wave H20-1 is locked before execution.

| GPU | Cell | Recipe | Quadmask | Timestep | Scope |
| ---: | --- | --- | --- | ---: | --- |
| 0 | `R1_Q2_T500_S0` | `R1_WinnerPreserve_LocalDPO` | `q2_strict_affected` | 500 | `S0_proj_out` |
| 1 | `R1_Q2_T300_S0` | `R1_WinnerPreserve_LocalDPO` | `q2_strict_affected` | 300 | `S0_proj_out` |
| 2 | `R2_Q2_T500_S0` | `R2_WinnerPreserve_LoserClip` | `q2_strict_affected` | 500 | `S0_proj_out` |
| 3 | `R1_Q1_T500_S0` | `R1_WinnerPreserve_LocalDPO` | `q1_object_only` | 500 | `S0_proj_out` |

Wave H20-2 is conditional only if Wave1 has no PASS but at least one MIXED-safe cell.

No VOR-Eval, hard comp, 10-step, long training, universal-adapter, final-SOTA, or third-backbone claim is allowed.
