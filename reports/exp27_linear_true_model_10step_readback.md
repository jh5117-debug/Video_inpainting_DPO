# Exp27 Linear True-Model 10-Step Readback

Date: 2026-06-25

## Git

- Branch: `research/exp27-paper-grounded-preference-study`
- HAL HEAD: `265bfe035eb260fba3001697486af720f1811f9d`
- Worktree status before milestone: new local script `exp27_paper_grounded_preference_study/scripts/run_exp27_linear_true_model_10step.py`
- Remote fetch: completed before this milestone.

## Files Read

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/49_exp27_paper_grounded_preference_study.md`
- `experiment_registry/exp27_paper_grounded_preference_study/status.md`
- Recent reports:
  - `reports/exp27_sdpo_true_model_forward_parity.md`
  - `reports/exp27_sdpo_true_model_distribution_scan.csv`
  - `reports/exp27_sdpo_true_model_summary.json`
  - `reports/exp27_linear_true_model_parity.md`
  - `reports/exp27_linear_true_model_parity.csv`
- Milestone code:
  - `exp27_paper_grounded_preference_study/scripts/run_exp27_linear_true_model_10step.py`
  - `exp27_paper_grounded_preference_study/scripts/run_exp27_true_model_objective_parity.py`
  - `exp27_paper_grounded_preference_study/code/official_parity.py`

## Source State

Already completed:

- `TRUE_MODEL_PARITY`
- `SDPO_TRUE_MODEL_32X4_SCAN_COMPLETE`
- `SDPO_TINY_STEP_ACTUAL_CHECK_PASSED`
- `LINEAR_TRUE_MODEL_PROBE_PASS`

Pending:

- Linear-DPO Frozen true-model 1/10-step.
- Linear-DPO EMA true-model 1/10-step.
- LocalDPO 24F adaptation.

Banned repeats / exclusions:

- Do not start RC-FPO.
- Do not start O0-O5 objective training.
- Do not report residual-proxy results as true-model parity.
- Do not report the prior Linear probe as 1/10-step training.

## Milestone Gate

This milestone may run only a micro 1/10-step true-model Linear-DPO gate on real
DiffuEraser Stage1 forward batches. It must record utility distribution,
margin, grad norm, policy/reference deltas, and EMA drift. It must not start
long training or RC-FPO.

## PAI State

- Hostname: `dsw-753014-85f54df947-bkp7h`
- PAI runtime code path:
  `/home/hj/runtime_code/H20_Video_inpainting_DPO_exp27_paper_study_run`
- PAI runtime HEAD observed before synchronization:
  `e708715e336dc2f2ec3fce97086f1963102c9aca`
- GPU state at readback: GPU0 lightly occupied by Exp26 step0 metrics; GPU1-7
  otherwise free at the time of the check.

## Planned Outputs

- PAI output root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp27_paper_grounded_preference_study/exp27_linear_true_model_10step_*`
- Reports:
  - `reports/exp27_linear_true_model_10step.md`
  - `reports/exp27_linear_frozen_10step.csv`
  - `reports/exp27_linear_ema_10step.csv`

