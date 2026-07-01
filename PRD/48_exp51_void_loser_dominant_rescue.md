# Exp51 VOID Loser-Dominant Rescue

Created: 2026-07-01T10:33:01+08:00
Branch: `research/exp51-void-loser-dominant-rescue-20260701`
Base: Exp50 HEAD `3aa65c53b2bd53e69e7a2d9528d9127a21849d66`

## Objective

Diagnose Exp50 loser-dominant 10-step behavior and test winner-preserving / local-DPO / SDPO-safe rescue recipes without modifying VOID official source, shared trainer, or `inference/metrics.py`.

## Milestone A

Status: `VOID_LOSER_DOMINANT_CONFIRMED`

Exp50 10-step margin growth is dominated by loser degradation rather than winner improvement. VOID remains a baseline / loser generator / adapter engineering candidate, not third adapter evidence.

## Safety

No VOR-Eval, no hard comp, no long training, no universal adapter, and no final SOTA claims.

## Exp51 Milestone B Quadmask Metrics - 2026-07-01T10:51:36+08:00

Status: `VOID_QUADMASK_METRICS_READY`. First-8-frame quadmask-aware audit shows 10-step local damage despite outside safety: affected_union delta PSNR -0.241561, overlap delta -0.423390, object_core delta -0.569839, outside_background delta 0.029117. Future rescue should prioritize object/object-core and affected boundary preservation while clipping loser gradients.

## Milestone C - Official SFT Parity Hard Audit

Status: `VOID_SFT_PARITY_EXPLAINED_ONLY`

The wrapper mirrors official target construction (`v_prediction` via scheduler velocity), inpaint conditioning, transformer forward, and mean MSE. Exact helper comparison remains unavailable because official `train.py` keeps the logic inside the Accelerator training loop.

## Milestone D - VOR Quadmask Ablation Data

Status: `VOID_QUADMASK_ABLATION_READY`

Built four quadmask variants for existing VOR-Train train4/heldout4 rows. All 8 visual sheets were opened. Q1 is clean object-only; Q2 is conservative affected; Q3 is broader and riskier on REAL texture/lighting spill.

## Milestone E - VOID-Native Kubric Diagnostic

Status: `VOID_NATIVE_KUBRIC_BLOCKED`

H20 lacks Kubric, PyBullet, Blender, HUMOTO, and Blender texture assets. Public GCS manifests are reachable but insufficient. No fake VOID-native data was created.

## Milestone F - Rescue Recipe Preregistration

Status: `VOID_RESCUE_RECIPES_PREREGISTERED`

R1/R2/R3/R4 are preregistered for one-step safety. R5 LoRA is gated and not default. No training was run before this preregistration commit.

## Milestone G - One-Step Rescue Grid

Status: `VOID_RESCUE_ONESTEP_BLOCKED`

R1-R4 train4 grid was attempted on H20 GPU0 but produced no checkpoint/report after a bounded micro window and was terminated. This is a runtime/runner blocker, not a recipe-negative result. H 10-step remains locked.
