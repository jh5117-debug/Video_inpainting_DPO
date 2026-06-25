# Exp27 Permission Recovery Readback - 2026-06-25

## Git

- branch: `research/exp27-paper-grounded-preference-study`
- HEAD: `2b1a491868c0c953ac1a4138a6374d3e8d3c8bb6`
- worktree status before milestone: clean except new permission-recovery reports

## Files Read

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/49_exp27_paper_grounded_preference_study.md`
- `experiment_registry/exp27_paper_grounded_preference_study/status.md`
- `reports/exp27_sdpo_real_distribution_scan.md`
- `reports/exp27_sdpo_real_distribution_scan.csv`
- `reports/exp27_true_model_forward_readback.md`
- `reports/pai_postmaintenance_asset_permission_matrix.csv`

## Completed State

- LocalDPO primitive/plumbing gates passed.
- SDPO constructed conflict case passed.
- Real residual-proxy scan completed, but it is not true model parity.

## Pending

- True DiffuEraser policy/reference SDPO forward parity.
- True Linear-DPO Frozen/EMA parity.
- Conditional LocalDPO 24F only after true SDPO/Linear completion.

## Banned Repeats

- Do not start RC-FPO.
- Do not label residual proxy as `TRUE_MODEL_PARITY`.
- Do not start O0-O5 or 50-step objective training in this milestone.

## PAI Permission State

- PAI host: `dsw-753014-85f54df947-bkp7h`
- DiffuEraser converted weights: readable/executable by `hj`.
- Exp27 experiments output: writable by `hj`.
- Exp27 autoresearch output: writable by `hj`.

Evidence:

- `reports/pai_postmaintenance_permission_recovery_final.md`
- `reports/pai_postmaintenance_permission_recovery_final.csv`
