# Exp25 Permission Recovery Readback - 2026-06-25

## Git

- branch: `research/exp25-vor-or-preference-data`
- HEAD: `48be2e9148714f7eb7e3045ba766807270247bbc`
- worktree status before milestone: clean except new permission-recovery reports

## Files Read

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/47_exp25_vor_or_preference_data.md`
- `experiment_registry/exp25_vor_or_preference_data/status.md`
- `reports/pai_postmaintenance_asset_permission_matrix.csv`
- `reports/pai_postmaintenance_system_audit.md`
- `reports/pai_postmaintenance_runtime_recovery.md`
- `reports/exp25_diffueraser_or_root_cause_matrix_20260625_status.md`

## Completed State

- Gate32 dense review completed.
- Root-cause 12-sample manifest exists and remains fixed.
- Previous blocker was DiffuEraser converted-weight read permission.

## Pending

- Post-permission DiffuEraser asset smoke.
- 12-sample root-cause matrix.

## Banned Repeats

- Do not expand Gate128.
- Do not start OR-DPO.
- Do not replace the fixed 12-sample root-cause manifest.
- Do not use VOR-Eval for training, selection, thresholding, or checkpoint choice.

## PAI Permission State

- PAI host: `dsw-753014-85f54df947-bkp7h`
- DiffuEraser converted weights: readable/executable by `hj`.
- `brushnet/config.json`: readable.
- `unet_main/config.json`: readable.
- Exp25 experiments output: writable by `hj`.
- Exp25 autoresearch output: writable by `hj`.

Evidence:

- `reports/pai_postmaintenance_permission_recovery_final.md`
- `reports/pai_postmaintenance_permission_recovery_final.csv`
