# Exp51 VOID Loser-Dominant Rescue Decision

Final status: `VOID_ADAPTER_ENGINEERING_CANDIDATE`

Also retained: `VOID_BASELINE_AND_LOSER_GENERATOR_READY`

VOID is still useful as a VOR-OR inference baseline and same-model loser generator. It is not third-backbone evidence.

## Required Answers

1. Was Exp50 10-step negative caused by loser-dominant DPO? Yes. Exp51 forensic confirmed margin growth was dominated by the loser branch: final winner_gap was small while loser_gap was much more negative, so vanilla DPO improved the objective mainly by degrading the loser.
2. Did winner-preserving / loser-clipped objective fix it? Not proven. Recipes were preregistered, but the R1-R4 one-step grid blocked before checkpoint/video evidence.
3. Did local-only preference help? Not evaluated; R1 LocalDPO blocked before checkpoint.
4. Did Linear-DPO v-prediction help? Not evaluated; R4 blocked before checkpoint.
5. Did SDPO-safe help? Not evaluated; R3 blocked before checkpoint.
6. Is proj_out-only sufficient? Still inconclusive. Exp50 proj_out passed zero-gap and one-step, but vanilla 10-step was negative and Exp51 rescue grid did not complete.
7. Is VOR-derived quadmask the blocker? Suspected but not proven. Quadmask audit showed local object/affected/boundary damage and Q3 broad affected spill on REAL rows; Q1/Q2 are better diagnostics.
8. Did VOID-native Kubric data help or is it blocked? Blocked. Missing Kubric/PyBullet/Blender/HUMOTO assets prevented native data generation.
9. Is VOID now third adapter evidence? No. VOID is not third adapter evidence.
10. Should we continue VOID, resume ROSE, or stop third-model search? Continue VOID only as a narrow engineering candidate with R1-only row0 smoke or Q1/Q2 ablation; in parallel, ROSE can be resumed if third-model search must continue. Do not run long VOID training.

## Evidence

- Forensic: `VOID_LOSER_DOMINANT_CONFIRMED`
- Quadmask metrics: local object/affected/boundary damage visible in Exp50 10-step; outside/full metrics can hide local degradation.
- SFT parity: `VOID_SFT_PARITY_EXPLAINED_ONLY`, sufficient for diagnostics but not byte-for-byte helper parity.
- Quadmask ablation data: `VOID_QUADMASK_ABLATION_READY`; Q1/Q2 are better rescue probes than broad Q3.
- Native Kubric: `VOID_NATIVE_KUBRIC_BLOCKED`.
- Rescue one-step: `VOID_RESCUE_ONESTEP_BLOCKED` before checkpoint, so no rescue recipe pass/fail and no 10-step unlock.

## Decision

Do not run 30/50/100-step VOID training. The next minimal VOID experiment is a narrower R1-only row0 smoke that proves checkpoint creation quickly, then recipe-by-recipe train4 heldout video evidence. If third-model search pressure is higher than VOID debugging value, resume ROSE while keeping VOID positioned as a baseline/loser-generator plus engineering candidate.
