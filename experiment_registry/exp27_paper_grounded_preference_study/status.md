# Exp27 Registry Status

PAPER_REVIEW_COMPLETE
EXACT_BASELINE_REPRODUCTION_IN_PROGRESS
NO_LONG_TRAINING
LOCALDPO_COMPAT_MASK_ONLY_PASSED

Primary candidate: RC-FPO.

Fallback candidate: ST-Pref.

LocalDPO official raw mask path remains blocked in the current environment, but Exp27 now has an isolated compatibility wrapper that passes mask-only deterministic probes. Real corruption parity and real DiffuEraser-batch SDPO/Linear parity remain pending.
## 2026-06-23 Overnight Autonomous Controller

- Status: `OVERNIGHT_CPU_PARITY_REFRESH_PASSED_REAL_BATCH_PENDING`.
- PAI controller runtime:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623`.
- CPU primitive parity refresh completed from a tracked git-archive snapshot.
- Real DiffuEraser-batch SDPO and Linear-DPO parity remain pending; no studies
  or RC-FPO training have started.

## 2026-06-23 GPU2 Real-Batch Parity

- Status: `REAL_BATCH_SDPO_AND_LINEAR_PARITY_PASSED`.
- SDPO real-batch parity: passed, finite objective, grad norm non-zero.
- Linear-DPO Frozen / EMA real-batch parity: passed, finite loss, grad norm
  non-zero, EMA max absolute difference `0.0`.
- Outputs:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623/exp27_sdpo_real_batch_parity`
  and
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623/exp27_linear_real_batch_parity`.
- Report: `reports/exp27_gpu2_real_batch_parity.md`.
- No long training was launched.
