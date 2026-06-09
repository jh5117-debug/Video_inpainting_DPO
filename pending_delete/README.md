# Pending Delete Area

This directory is the holding area for legacy experiment assets that should no
longer appear in the active code/registry structure, but are not deleted yet.

Policy:

- Do not run scripts from this directory for current experiments.
- Do not use registry entries here as current conclusions.
- Restore from here only if a legacy result must be audited or reproduced.
- Physical deletion requires a separate explicit review.

Current active experiment ledger is maintained in:

- `experiment_registry/current_active.md`
- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`

Moved here on 2026-06-09:

- Old generic Exp8 region-loss registry entries superseded by Exp8a/Exp8c and
  Exp9/Exp10/Exp11.
- Old Exp9 D3 gate/nolose/nocomp registry entries whose names conflict with
  the new Exp9 log-ratio experiment.
- Old Exp9 gate launch scripts superseded by `scripts/launch_exp09_10_11_pai.sh`.
