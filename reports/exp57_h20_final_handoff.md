# Exp57 H20 Adaptive Transition Final Handoff

Status: `EXP57_H20_ONESTEP_NEGATIVE`

Branch: `research/exp57-void-adaptive-transition-h20-20260701`

Core base: `83e1bc7a5e5da231251d9c74f33a2ec49c8319f4`

## Scope

H20 ran one-step only on Q2/T500/S0 using the shared Exp57 adaptive transition-safe loss implementation. No 10-step, long training, VOR-Eval, hard comp, shared trainer edit, VOID official source edit, or `inference/metrics.py` edit was performed.

## Cells

| Cell | Status | Full PSNR delta | Object PSNR delta | Overlap PSNR delta | Affected PSNR delta | Boundary PSNR delta | Outside PSNR delta | Visual | Selected scale | Lambda loser |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| ATS0_Q2_T500_S0 | NEGATIVE | -0.113495 | -0.598885 | -0.593289 | -0.213889 | -0.124392 | -0.007378 | 1 better / 0 tie / 3 worse | 1.0000 | 0.046926 |
| ATS_STRICT_Q2_T500_S0 | NEGATIVE | -0.094041 | -0.473428 | -0.581142 | -0.146322 | -0.081617 | -0.005963 | 1 better / 0 tie / 3 worse | 1.0000 | 0.018770 |
| ATS_HALFLR_Q2_T500_S0 | NEGATIVE | -0.104624 | -0.773073 | -0.707366 | -0.160285 | -0.150837 | -0.002763 | 0 better / 0 tie / 4 worse | 1.0000 | 0.018770 |
| ATS_NODPO_Q2_T500_S0 | NEGATIVE | -0.094423 | -0.786377 | -0.563998 | -0.233663 | -0.123053 | -0.001418 | 0 better / 0 tie / 4 worse | 1.0000 | 0.000000 |

Best H20 diagnostic cell: `ATS_STRICT_Q2_T500_S0`.

## Required Questions

1. Did adaptive transition safety reduce overlap / affected / boundary regression?

No. All four cells still regressed overlap, affected, and boundary on heldout4. `ATS_STRICT_Q2_T500_S0` was least bad overall, but still violated the one-step PASS gate.

2. Did adaptive safe-lambda reduce loser dominance?

Partially as a diagnostic: `ATS_STRICT` and `ATS_HALFLR` reduced lambda to `0.018770`, and `ATS_NODPO` forced lambda to `0.0`. However, heldout damage persisted even with no loser DPO, so the dominant blocker remains transition-region damage under the policy update.

3. Did backtracking choose full update or smaller scale?

Full scale `1.0` was selected for every cell. The train4 finite-difference checks did not reject the updates.

4. Did any cell reject update as unsafe?

No. No update was rejected.

5. Which H20 cell is best?

`ATS_STRICT_Q2_T500_S0`, because it had the least negative full/object/affected/boundary mean deltas among the adaptive cells.

6. Is any H20 cell one-step PASS?

No. Every H20 Exp57 adaptive cell is NEGATIVE under the original one-step gate.

7. Should aggregator consider H20 candidate?

Yes, but only as a negative/mixed diagnostic. It should not unlock 10-step.

## Decision

`EXP57_H20_ONESTEP_NEGATIVE`

VOID remains a VOR-OR inference baseline, same-model loser-generator candidate, and adapter-engineering candidate. This H20 lane does not provide third-backbone adapter evidence.
