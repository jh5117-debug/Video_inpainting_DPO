## 2026-06-04 Exp7 Small-Mask / Prior Reset

This update supersedes any plan to expand D3/YouTube-VOS DPO before Exp7 is fixed.

Required facts:

- YouTube-VOS / D3 work must use the fine-tuned SFT-48000 DiffuEraser weights, not an ordinary base.
  - PAI path: `/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000`
  - HAL local reference: `/home/hj/Video_inpainting_DPO/weights/diffuEraser/converted_weights_step48000`
  - H20 confirmed path: `/home/nvme01/H20_Video_inpainting_DPO/weights/diffuEraser/converted_weights_step48000`
  - H20 path must be rechecked; SSH was blocked in this turn.
- Every experiment must have an `experiment_registry/<exp>/` folder with config, commands, paths, status, metric, qualitative, and dpo_diag notes.
- Every DPO experiment needs `dpo_diagnostics.csv`; missing diagnostics are marked `MISSING_DPO_DIAG`.
- Current lineage is: Exp4 failed fullmask data gate; Old Exp5 plain DPO collapse; New Exp5 comp + winner anchoring improves but is not final; New Exp6 no-comp + same winner anchoring is not plain Exp6; Exp7 changed task to partial mask but is suspicious; Exp8/Exp9 are target-domain diagnostics only.
- Fullmask/video-generation settings should not rely on ProPainter prior. Partial-mask video inpainting should use DiffuEraser/OR ProPainter prior.
- VideoDPO partial-mask masks should be reduced to 15%-20% for Exp7-fix; do not continue old large-mask D2 for new Exp7 gates.
- Train may use YouTube-VOS, but final target eval should use DAVIS at `/mnt/workspace/hj/nas_hj/data/external/davis_432_240`.

Current Exp7 audit:

- `tools/eval_generated_loser_partialmask_model.py` appears to use a direct pipeline path with masked winner frames and masks, without `inference/run_OR.py` or `--propainter_model_dir`.
- This makes the current Exp7 eval likely no-prior and may explain why SFT/base videos also look bad.
- Report: `reports/exp7_prior_mask_audit.md`.

Next planned gates after small-mask/prior data exists:

- PAI: `exp07_fix_smallmask_prior_wingap_lose025_stage1_gate1500`.
- H20: `exp07_fix_smallmask_prior_wingap_nolose_stage1_gate1000`.

Do not run DPO Stage2, VBench for inpainting, or long D3 sweeps until this sanity gate is reviewed.

# Experiment Registry And DPO-Diagnostics Audit

Updated: 2026-06-04

## Why This Exists

The early PRD required every experiment to have its own folder and complete artifact record. The project now has a formal `experiment_registry/` so Exp4, Old Exp5, New Exp5, New Exp6, Exp7, Hybrid, Exp8, and Exp9 are no longer mixed together in prose.

## Registry Contract

Each experiment folder contains:

- `README.md`
- `paths.yaml`
- `commands.md`
- `dpo_diag_summary.md`
- `metrics_summary.md`
- `qualitative_summary.md`
- `status.md`

The registry stores paths only. It must not copy videos, checkpoints, datasets, or weights into git.

## Corrected Experiment Lineage

| Registry ID | Correct name | Status summary |
| --- | --- | --- |
| Exp4 | `exp04_fullmask_loser_failed_gate` | Failed full-mask loser data quality gate / diagnostic only. |
| Old Exp5 | `exp05_old_d2_comp_plain_failed` | D2 comp data-only run with unsafe/unanchored objective; collapse evidence needs PAI dpo_diag recovery. |
| New Exp5 | `exp05_new_d2_comp_wingap_lose025` | D2 comp with winner anchoring; improved but not final; PAI dpo_diag pending. |
| New Exp6 | `exp06_new_d2_nocomp_wingap_lose025` | D2 no-comp + winner anchoring. This is not plain Exp6. H20 dpo_diag found. |
| Exp7 | `exp07_d2_partialmask_task_wingap` | First task-aligned partial-mask DPO gate; metrics show Stage1 improves, Stage2 regresses; PAI dpo_diag pending. |
| Hybrid | `exp07_dpoS1_sftS2_hybrid` | No-training hybrid checkpoint audit/eval; negative / not final. |
| Exp8 | `exp08_d3_comp_regionloss_gate` | Planned target-domain region-loss diagnostic; not a completed success. |
| Exp9 comp | `exp09_d3_comp_stage1_gate` | PAI D3-comp Stage1 gate; ckpt500 best by pasted metrics; dpo_diag pending. |
| Exp9 nocomp | `exp09_d3_nocomp_stage1_gate` | H20 D3-nocomp Stage1 gate; dpo_diag and target eval found. |
| Exp9 no-lose | `exp09_d3_comp_nolose_gate` | H20 D3-comp no-lose gate; dpo_diag found, target eval pending. |

## DPO-Diagnostics Rule

`dpo_diagnostics.csv` is mandatory evidence for every DPO training experiment. A result with metric/qualitative evidence but no diagnostic CSV is marked incomplete until PAI/H20 paths are recovered.

The key fields tracked are:

- `dpo_loss`
- `implicit_acc`
- `mse_w`, `ref_mse_w`, `mse_l`, `ref_mse_l`
- `win_gap`, `lose_gap`
- `winner_gap_reg`
- `mse_w_over_ref_mse_w`, `mse_l_over_ref_mse_l`
- `sigma_term`
- `kl_divergence`
- `loser_dominant_ratio`
- `grad_norm`

## Current Aggregate DPO-Diag Findings

Generated files:

- `reports/all_experiments_dpo_diag_summary.csv`
- `reports/all_experiments_dpo_diag_summary.md`

Current local/H20-backed diagnostics:

- New Exp6 Stage1: found, but summary labels show saturation / loser-dominant risk in long training.
- Exp9 nocomp Stage1: found, also shows loser-dominant/collapse-risk signals despite target-domain gate completion.
- Exp9 no-lose Stage1: found, loser-dominant shortcut is reduced by `lose_gap_weight=0.0`, but implicit accuracy can still saturate; target eval is still needed.

Missing or pending diagnostics:

- Exp1/Exp2/Exp3 historical runs.
- Exp4, because it failed as a data quality gate.
- Old Exp5 and New Exp5 PAI diagnostics.
- Exp7 PAI diagnostics.
- Exp9 PAI clean comp diagnostics.

## PAI Manual Audit Requirement

PAI must be scanned with fixed globs only. Do not recursively scan `/mnt/workspace` or the whole NAS. The manual command block in the final response writes:

- `reports/pai_experiment_registry_audit.md`
- `reports/pai_experiment_registry_paths.csv`

After those files are returned, the registry should be updated so Old Exp5, New Exp5, Exp7, and Exp9-comp no longer remain `MISSING_DIAG` if the CSVs exist.

## PPT Correction Boundary

The revised weekly deck must say:

- Old Exp5 collapsed: ranking was satisfied but visual quality broke.
- New Exp5 added winner anchoring: improved guardrail, not final success.
- New Exp6 is no-comp + winner anchoring, not plain Exp6.
- Exp8/Exp9 target-domain inpainting uses `inference/metrics.py` wrappers, not VBench.
- Final claims need metric + qualitative + dpo_diag.
