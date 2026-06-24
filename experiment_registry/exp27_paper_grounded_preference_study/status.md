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

## 2026-06-24 Nontrivial Parity and LocalDPO Smoke

- Status: `NONTRIVIAL_PARITY_AND_LOCALDPO_SMOKE_PASSED`.
- Controller run:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2`.
- SDPO nontrivial conflict case passed with `lambda_safe=0.314453125 < 1`.
- Linear-DPO Frozen / EMA multi-step parity passed, `ema_max_abs_diff_max=0.0`.
- LocalDPO six-video corruption pair and original loss 1/10-step smoke passed.
- Official LocalDPO mask digest remains blocked by missing official code file.
- RC-FPO was not started.
- Report: `reports/exp27_nontrivial_parity_and_localdpo_smoke_20260624.md`.

## 2026-06-24 Distribution Scan and Official LocalDPO Path Fix

- Status:
  - `NONTRIVIAL_SDPO_PARITY_PASSED`
  - `LINEAR_MULTISTEP_PARITY_PASSED`
  - `LOCALDPO_SMOKE_PASSED`
  - `SDPO_REAL_RESIDUAL_PROXY_SCAN_COMPLETE`
  - `FAITHFUL_LOCALDPO_OFFICIAL_MASK_DIGEST_PASSED`
  - `OBJECTIVE_STUDY_PENDING`
  - `RCFPO_NOT_STARTED`
- SDPO residual-proxy records: `128`.
- `lambda_safe < 1` ratio: `0.4453125`.
- `lambda_safe` min/mean/max: `0.2246925 / 0.8942396 / 1.0`.
- Official LocalDPO mask digest now passes from the commit-suffixed PAI cache path.
- Still pending: full policy-forward distribution scan, LocalDPO four-grid objective runs, and O0-O5 objective study.
- Reports:
  - `reports/exp27_sdpo_real_distribution_scan.md`
  - `reports/exp27_localdpo_official_path_fix.md`

## 2026-06-24 True-Model Forward Readback

- Status:
  - `TRUE_MODEL_FORWARD_READBACK_COMPLETE`
  - `SDPO_REAL_RESIDUAL_PROXY_ONLY`
  - `OBJECTIVE_STUDY_PENDING`
  - `RCFPO_NOT_STARTED`
- Existing 128-record SDPO distribution scan is still proxy-only: it uses real
  video residuals but no DiffuEraser policy/reference forward.
- Next required gate is true Stage1 policy/reference forward parity.
- Report: `reports/exp27_true_model_forward_readback.md`.
## 2026-06-25 PAI Pre-Maintenance Persistence

Status: `BLOCKED_NAS_PERMISSION`

The cross-track persistence gate was attempted before new Exp27 GPU work.
`hj` cannot write to the requested NAS project root, so no new true-model SDPO
forward, Linear-DPO, LocalDPO 24F, or objective-study task was launched.

Report: `reports/pai_premaintenance_output_persistence.md`
