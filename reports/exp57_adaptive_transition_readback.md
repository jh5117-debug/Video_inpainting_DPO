# Exp57 Adaptive Transition Readback

Status: `EXP57_READBACK_DONE`

## Source Branches

- Base: `origin/research/exp55-void-crosslane-aggregator-20260701`
- Exp56-H20 reference: `origin/research/exp56-void-region-safe-h20-20260701`

## Answers

1. Which prior recipes were mixed? Exp53B H20 R1/R2 Q2/T500, Exp54 PAI R3/R4 diagnostics, and Exp56-H20 R5/R5_HALF were all mixed-only. No one-step PASS was available.
2. Did R5 remove loser-dominant behavior? Yes. Exp56-H20 `R5_Q2_T500_S0` and `R5_HALF_Q2_T500_S0` had `loser_contribution_ratio = 0.0`.
3. Which regions still regressed after R5? Overlap, affected, and boundary. `R5_Q2_T500_S0` had overlap PSNR -0.153271, affected PSNR -0.084209, and boundary PSNR -0.047360.
4. Why is fixed loser suppression insufficient? Because even with loser contribution removed, the object-local update still spills into transition regions.
5. Why is transition-region safety required? The active blocker is now `TRANSITION_REGION_DAMAGE_UNDER_OBJECT_LOCAL_UPDATE`: object and outside can improve while overlap / affected / boundary fail the one-step gate.
6. Why should Exp57 be VOID-only for now? The issue is specific to VOID's quadmask, v-prediction wrapper, cached train4/heldout4 setup, and same-model loser evidence. DiffuEraser and VideoPainter should not inherit this experimental loss.
7. Why should DiffuEraser / VideoPainter current best loss remain unchanged? They are the current positive adapter evidence; Exp57 is a VOID rescue experiment with an unproven transition-safe objective.

## Readback Metrics

| cell | full | object | overlap | affected | boundary | outside | visual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R5_Q2_T500_S0 | +0.013859 | +0.956095 | -0.153271 | -0.084209 | -0.047360 | +0.047483 | 0 better / 2 tie / 2 worse |
| R5_HALF_Q2_T500_S0 | +0.012121 | +0.667133 | -0.139584 | -0.113662 | -0.069402 | +0.047722 | 0 better / 2 tie / 2 worse |

## Decision

Exp57 should implement `void_adaptive_transition_safe_dpo_v0` with adaptive loser safe-lambda, transition-region finite-difference safety, and backtracking/no-update handling. No 10-step is allowed.
