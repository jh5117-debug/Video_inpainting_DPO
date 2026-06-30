# Exp50 PAI VOID Adapter Feasibility

This registry tracks VOID as a possible third adapter candidate.

Milestone A status: `EXP50_VOID_READBACK_COMPLETED_WITH_NAS_PERMISSION_CAVEAT`.

No asset download or training has been run. VOID public sources indicate stronger adapter feasibility than ROSE because they expose inference, data-generation, quadmask format, and training scripts.

## Milestone B3 update - VOID_WEIGHTS_READY

- Time: 2026-06-30T14:04:06.075339+08:00
- Status: `VOID_WEIGHTS_READY`
- Evidence: `reports/exp50_void_weight_relay_ingest.md`
- Relay SHA match: yes, 52 / 52 files, missing 0, mismatch 0.
- Safety: no training, no inference, no GPU, no VOID positive claim.

## Milestone C update - VOID_ENV_PARTIAL

- Time: 2026-06-30T14:20:58.202107+08:00
- Status: `VOID_ENV_PARTIAL`
- Evidence: `reports/exp50_void_env_smoke.md`
- Imports: 44 pass, 0 fail.
- CUDA smoke: no failures; small matmul/backward finite.
- Caveat: exact official pins are not matched; heavyweight CUDA packages were not reinstalled.
- Safety: no training, no inference, no full 5B model load, no VOID positive claim.

## Milestone D update - VOID_TRAINING_FORWARD_HEAVY_BUT_POSSIBLE

- Time: 2026-06-30T14:24:40.037508+08:00
- Status: `VOID_TRAINING_FORWARD_HEAVY_BUT_POSSIBLE`
- Evidence: `reports/exp50_void_trainable_forward_audit.md`
- Finding: official trainable forward exists, but default training is heavy transformer fine-tuning, not out-of-box LoVI-DPO.
- Safety: no training, no inference, no official source modification, no VOID positive claim.

## Milestone E update - VOID_VOR_QUADMASK_GATE8_READY

- Time: 2026-06-30T14:40:28.047764+08:00
- Status: `VOID_VOR_QUADMASK_GATE8_READY`
- Evidence: `reports/exp50_void_vor_quadmask_adapter.md` and `reports/exp50_void_vor_quadmask_visual_review.csv`
- Gate8: 8 rows, REAL/BLENDER 4/4, scene overlap False.
- VOR-Eval excluded: True
- Safety: no training, no inference, no hard comp, no VOID positive claim.
- Next gate: official inference smoke remains blocked until environment status is `VOID_ENV_READY`; current C status is `VOID_ENV_PARTIAL`.
