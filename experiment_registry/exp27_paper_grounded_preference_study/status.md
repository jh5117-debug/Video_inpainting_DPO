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

## 2026-06-25 PAI Pre-Maintenance Persistence Resolved

Status:

- `PAI_PREMAINTENANCE_PERSISTENCE_PASSED`
- `TRUE_MODEL_FORWARD_READBACK_COMPLETE`
- `SDPO_REAL_RESIDUAL_PROXY_ONLY`
- `OBJECTIVE_STUDY_PENDING`
- `RCFPO_NOT_STARTED`

Cross-track persistence now passed:

- Exp25 Gate32 dense review: `99` files / `66982608` bytes / inventory OK /
  SHA256 OK.
- Exp26 Gate64 official generation: `14408` files / `8405904095` bytes /
  inventory OK / SHA256 OK.

Completion markers are present under:

`/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/`

## 2026-06-25 SDPO Real Residual-Proxy Distribution Scan

Status:

- `SDPO_REAL_RESIDUAL_PROXY_SCAN_COMPLETE`
- `OBJECTIVE_STUDY_PENDING`
- `RCFPO_NOT_STARTED`

Records: `128` from `32` real preference rows.

`lambda_safe < 1` ratio: `0.4453125`.

lambda min/mean/max: `0.2246925 / 0.8942396 / 1.0`.

Unsafe tiny-step winner-change rate: `0.0`.

This remains residual-proxy only and cannot promote RC-FPO or objective-study
training.

Reports:

- `reports/exp27_sdpo_real_distribution_scan.md`
- `reports/exp27_sdpo_real_distribution_scan.csv`
- `reports/exp27_sdpo_real_distribution_scan_summary.json`

## 2026-06-25 PAI Post-Maintenance Permission Recovery

Status:

- `PAI_POSTMAINTENANCE_PERMISSIONS_RECOVERED`
- `BLOCKER_RESOLVED`
- `TRUE_MODEL_SDPO_READY_TO_RUN`
- `OBJECTIVE_STUDY_PENDING`
- `RCFPO_NOT_STARTED`

Confirmed on PAI host `dsw-753014-85f54df947-bkp7h` as user `hj`:

- DiffuEraser converted weights: readable/executable.
- Exp27 NAS experiment output: writable.
- Exp27 NAS autoresearch output: writable.

Reports:

- `reports/exp27_permission_recovery_readback.md`
- `reports/pai_postmaintenance_permission_recovery_final.md`
- `reports/pai_postmaintenance_permission_recovery_final.csv`

## 2026-06-25 True DiffuEraser Policy/Reference SDPO Gate

Status:

- `TRUE_MODEL_PARITY`
- `SDPO_TRUE_MODEL_32X4_SCAN_COMPLETE`
- `SDPO_TINY_STEP_ACTUAL_CHECK_PASSED`
- `LINEAR_TRUE_MODEL_PROBE_PASS`
- `LINEAR_TRUE_MODEL_1_10_STEP_PENDING`
- `LOCALDPO_24F_PENDING`
- `RCFPO_NOT_STARTED`

Records: `256` true model forwards over S0/S1, with `128` S1 records.
S1 `lambda_safe < 1` count: `32` (`0.25`). Lambda max abs diff and SDPO loss
max abs diff against the extracted official SDPO helper are both `0.0`.
Output-gradient cosine min is `0.9999998807907104`. Eight actual tiny-step cases
passed (`4` lambda<1 and `4` lambda=1), with max reference grad norm `0.0` and
max policy parameter delta norm `0.0009144449931267993`.

Reports:

- `reports/exp27_sdpo_true_model_forward_parity.md`
- `reports/exp27_sdpo_true_model_distribution_scan.csv`
- `reports/exp27_sdpo_true_model_summary.json`
- `reports/exp27_sdpo_true_model_tiny_step_cases.csv`
- `reports/exp27_linear_true_model_parity.md`
- `reports/exp27_linear_true_model_parity.csv`
