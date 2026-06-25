# Exp26 Permission Recovery Readback - 2026-06-25

## Git

- branch: `research/exp26-videopainter-dpo-v2`
- HEAD: `7f85ed989ea33b8babdb1d2db697a92c99279f4f`
- worktree status before milestone: clean except new permission-recovery reports

## Files Read

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/48_exp26_videopainter_dpo_v2.md`
- `experiment_registry/exp26_videopainter_dpo_v2/status.md`
- `reports/exp26_gate64_final_readiness.md`
- `reports/exp26_gate64_duplicate_source_deep_audit.md`
- `reports/exp26_gate64_source_repair_readback.md`
- `reports/pai_postmaintenance_permission_recheck_20260625.md`

## Completed State

- Gate64 formal-valid sources: `64/64`.
- Existing evidence review: `64/64`.
- Candidate pool: `55` eligible rows.
- Primary-32 draft manifest exists.

## Pending

- Strict temporal mp4 playback review for all 64 rows.
- Final primary-32 lock after temporal review.
- VideoPainter DPO L0/L1, 10-step, and conditional 50-step.

## Banned Repeats

- Do not regenerate the completed 64 Gate64 outputs.
- Do not replace sources or seeds to hide failed cases.
- Do not start 100+ step or long training.

## PAI Permission State

- PAI host: `dsw-753014-85f54df947-bkp7h`
- Exp26 experiments output: writable by `hj`.
- Exp26 autoresearch output: writable by `hj`.
- Gate64 official output root: readable/writable by `hj`.

Evidence:

- `reports/pai_postmaintenance_permission_recovery_final.md`
- `reports/pai_postmaintenance_permission_recovery_final.csv`
