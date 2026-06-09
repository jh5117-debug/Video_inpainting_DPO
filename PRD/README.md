# Video Inpainting DPO PRD Index

Start here:

0. `../experiment_registry/current_active.md` - compact current experiment ledger and naming policy.
1. `pai_audit_current_state.md` - current environment and artifact audit.
2. `00_current_status.md` - completed experiments and current best settings.
3. `01_experiment_matrix.md` - active and planned experiment matrix.
4. `02_pai_runbook.md` - PAI checks and launch conventions.
5. `03_data_generation_plan.md` - loser generation plan and manifest schema.
6. `04_metrics_and_diagnostics.md` - metrics, DPO diagnostics, and loss-region notes.
7. `05_paths_and_artifacts.md` - data/weight/output path policy.
8. `07_exp5_exp6_winner_anchored_rerun.md` - winner-anchored reruns and Exp7 partial-mask gate decision trail.
9. `08_experiment_results_20260602.md` - compact ledger of completed/running experiments through Exp7-PM-Gate1500.
10. `data_and_weight_assets.md` - PAI asset roots and generated loser storage.
11. `data_generation_manifest_schema.md` - JSONL schema for generated loser manifests.
12. `dpo_diagnostics_and_metrics_plan.md` - future training diagnostics and metric boundaries.
13. `pai_asset_readiness_report.md` - PAI-generated asset readiness report template/output.
14. `pai_generated_data_summary.md` - current PAI generated-loser readiness and smoke-test gate.
15. `videodpo_canonical_data_setting.md` - generated on PAI by the single-sample smoke tool; records the canonical VideoDPO H/W/T sampling contract.

`archive/` contains historical plans and handoff notes. Archive files are useful context, but they do not define the current experiment plan unless the information has been migrated into the active PRDs above.

`../pending_delete/` is the non-destructive holding area for legacy experiment
files that should not be part of the active code/registry structure.
