# Exp52 VOID Rescue Decision

Final status: `VOID_ADAPTER_ENGINEERING_CANDIDATE`

Exp52 fixed the Exp51 slow-forward/no-checkpoint blocker enough to run a bounded one-step grid. R1 row0 smoke passed, Wave1 produced seven one-step checkpoints, and `R1_Q0_T500_S0` generated heldout4 Step0/Step1 video evidence.

## Answers

1. Was Exp50 10-step negative caused by loser-dominant DPO?
   Yes. Exp51 confirmed the loser-dominant margin mechanism; Exp52 showed R1 can suppress the intended train-side loser push.

2. Did R1 winner-preserving local DPO reduce loser dominance?
   Yes directionally. R1 row0 loser contribution ratio was 0.0143; Wave1 R1 train effective loser gap was 0.0 by design. Heldout remained mixed.

3. Did R2 loser clipping help?
   Inconclusive. R2 produced checkpoints but only forward-level mixed diagnostics; no full heldout video gate was completed.

4. Did R3 SDPO-safe help?
   Inconclusive. R3 produced checkpoints but only forward-level mixed diagnostics; no full heldout video gate was completed.

5. Did R4 LinearDPO-vPrediction help?
   Inconclusive. R4_Q0 produced a checkpoint with mixed forward diagnostics; R4_Q2 was skipped because GPU7 had an unrelated external process.

6. Which quadmask variant is safest?
   Q0 current is the only variant with complete Step0/Step1 video evidence in Exp52. It was safe globally/outside but not safe enough in affected/overlap. Q2 strict affected remains the next targeted diagnostic candidate, not a proven winner.

7. Is proj_out sufficient?
   Not proven. `proj_out` is sufficient for checkpoint/reload/finite one-step mechanics, but not sufficient yet for heldout positive evidence.

8. Is LoRA needed?
   Not proven. LoRA was not escalated because the S0 one-step evidence was mixed and affected-region risk remains unresolved.

9. Did any recipe produce one-step PASS?
   No. `R1_Q0_T500_S0` was `VOID_RESCUE_ONESTEP_MIXED`, not PASS.

10. Did any recipe produce 10-step positive/promising?
    No. 10-step was not run because one-step PASS was not achieved.

11. Did VOID become third adapter evidence?
    No.

12. If not, exact blocker.
    `VOID_RESCUE_ONESTEP_MIXED`: winner-preserving local DPO improved full/object/boundary/outside metrics but regressed affected/overlap and had tie-heavy visual evidence, so 10-step was locked.

13. Should we continue VOID, resume ROSE, or stop third-model search?
    Continue VOID only as a narrow adapter-engineering candidate: next minimal experiment is targeted one-step evidence for `R1_Q2_T500_S0` and one Q1/Q2/T300 nearby ablation. Do not long-train VOID. ROSE can be resumed separately if its feasibility gate is cleaner.

## Key R1_Q0 Heldout Deltas

- full PSNR: 0.01562676520194195
- object PSNR: 1.025830221887892
- overlap PSNR: -0.11671521679298635
- affected PSNR: -0.11865014078788594
- boundary PSNR: 0.1608491675666972
- outside PSNR: 0.04482370447721884
- visual: 0 better / 3 tie / 1 worse

No VOR-Eval, hard comp, long training, or final SOTA claim was used.
