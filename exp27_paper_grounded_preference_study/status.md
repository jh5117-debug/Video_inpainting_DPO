# Exp27 Status

- PAPER_REVIEW_COMPLETE
- EXACT_BASELINE_REPRODUCTION_IN_PROGRESS
- NO_LONG_TRAINING
- LOCALDPO_COMPAT_MASK_ONLY_PASSED
- SDPO_LINEAR_REAL_BATCH_PARITY_PENDING

## Gate Notes

- LocalDPO official random-mask raw path is blocked by a matplotlib ARGB/RGB compatibility issue in the official cached code path.
- Exp27 isolated compatibility wrapper passes mask-only deterministic probes without editing the official clone.
- Diffusion-SDPO scalar safe-lambda toy parity passed exactly.
- Linear-DPO utility and EMA toy parity passed exactly.
- No long training has been launched from Exp27.

## 2026-06-23 LocalDPO Fusion Primitive

- Added isolated LocalDPO latent fusion / progressive outside reinjection helper.
- Unit tests verify outside-latent preservation and separate task/corruption/restoration masks.
- CPU parity script now writes `localdpo_full_parity.json`.
- Real DiffuEraser-batch SDPO and Linear-DPO parity remain pending; no studies or RC-FPO runs started.

Status: `LOCALDPO_FUSION_PRIMITIVE_READY_REAL_BATCH_PENDING`.

## 2026-06-23 PAI Official Cache Sync

- Pinned Local-DPO, Diffusion-SDPO, and Linear-DPO caches synced to PAI.
- Compatibility symlink root: `/mnt/nas/hj/video_dpo_paper_code_cache/repos`.
- PAI CPU primitive parity passed for LocalDPO mask, LocalDPO latent fusion,
  SDPO lambda, Linear-DPO loss, and Linear-DPO EMA.
- Real DiffuEraser batch parity remains pending.

Status: `PAI_CPU_PRIMITIVE_PARITY_PASSED_REAL_BATCH_PENDING`.

## 2026-06-23 Overnight Autonomous Controller

- Status: `OVERNIGHT_CPU_PARITY_REFRESH_PASSED_REAL_BATCH_PENDING`.
- PAI controller runtime:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623`.
- CPU primitive parity refresh completed on PAI from the immutable runtime
  snapshot.
- Refreshed parity outputs are under:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623/exp27_cpu_parity_refresh`.
- Real DiffuEraser-batch SDPO and Linear-DPO parity are still pending a real
  GPU implementation/run; no Data Study, Objective Study, LocalDPO four-cell
  experiment, RC-FPO run, or long training has started.
