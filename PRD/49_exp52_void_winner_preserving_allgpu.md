# Exp52 VOID Winner-Preserving All-GPU Rescue

Branch: `research/exp52-void-winner-preserving-allgpu-20260701`

Status: `EXP52_ALL_GPU_READY`

Goal: fix Exp51 slow-forward/no-checkpoint blocker, cache VOID-native training inputs, run bounded winner-preserving rescue objectives, and only unlock 10-step if one-step video/metric gates pass.

Forbidden: VOR-Eval for training/filtering/tuning, hard comp, long training, official VOID source edits, shared trainer edits, `inference/metrics.py` edits, universal adapter claims, final SOTA claims, and VOID third-adapter evidence without a true heldout-positive micro gate.

## Milestone A - Readback and GPU Audit

Status: `EXP52_ALL_GPU_READY`

H20 GPU0-7 are free and no stale Exp50/51/52 GPU processes were killed. Exp52 will not run long training before cache and row0 smoke gates pass.

## Milestone B - Slow-Forward Forensic And Cache

Status: `VOID_CACHE_PARITY_EXPLAINED`

Generated 32 train4/heldout4 Q0-Q1-Q2-Q3 cache rows. The original parity failure was a helper precision issue on scalar reference losses, not a cache tensor mismatch.

## Milestone C - R1 Row0 Smoke

Status: `VOID_R1_ROW0_SMOKE_PASS`

R1 row0 produced checkpoint `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp52_void_winner_preserving_allgpu/r1_row0_smoke/checkpoints/r1_q0_t500_proj_out_row0_step1.pt`; strict reload and forward were finite. Winner gap post was positive and loser contribution ratio was 0.0143.

## Milestone D - All-GPU One-Step Rescue Preregistration

Status: `EXP52_RESCUE_GRID_PREREGISTERED`

Wave 1 preregisters R1/R2/R3/R4 on Q0/Q2 at T500 with S0 proj_out. Wave 2 is conditional. GPU7 is only used if free at execution audit.

## Milestone E Wave1 One-Step Rescue Grid - 2026-07-01T14:53:24+08:00

Status: `VOID_RESCUE_ONESTEP_MIXED`

Wave1: 7/8 one-step checkpoints produced; R4_Q2 skipped due unrelated GPU7 process. R1_Q0 video-evaluated: full 0.015627, object 1.025830, affected -0.118650, boundary 0.160849, outside 0.044824; visual 0 better / 3 tie / 1 worse; 10-step locked.

Decision: do not run 10-step from this result. R1_Q0 reduced loser dominance but did not deliver affected-region-safe visual improvement. Wave2 or a narrower Q2/Q1 diagnostic can be considered later, but Exp52 does not promote VOID to adapter evidence.

## Milestone F Wave2 / Scope Decision - 2026-07-01T14:55:45+08:00

Status: `EXP52_WAVE2_DEFERRED_NO_ONESTEP_PASS`. Wave2, LoRA escalation, and 10-step are not run because the only full video-evaluated Wave1 cell is mixed, not PASS.
