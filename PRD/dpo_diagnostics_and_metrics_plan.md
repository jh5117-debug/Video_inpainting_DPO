# DPO Diagnostics And Metrics Plan

This is a plan for later training runs. The current asset-preparation phase
does not start training and does not modify loss math.

## Reuse Existing Diagnostics

Search and reuse existing VideoDPO/DiffuEraser diagnostics before adding any
new helper:

- `implicit_acc`
- `win_gap`
- `lose_gap`
- `mse_w`
- `ref_mse_w`
- `mse_l`
- `ref_mse_l`
- `loser_dominant_ratio`
- `sigma_term`
- `grad_norm`
- `dpo_loss`
- `sft_reg_loss`
- `total_loss`

Future launchers should expose:

```text
--enable_dpo_diag true
--dpo_diag_log_every 10
--dpo_diag_save_csv true
--dpo_diag_save_wandb true
```

Outputs should include stdout/log lines, CSV, W&B if enabled, and
`run_manifest.json`.

Current implementation status:

- Stage 1 and Stage 2 training expose `--enable_dpo_diag`, `--dpo_diag_log_every`, `--dpo_diag_save_csv`, and `--dpo_diag_save_wandb`.
- Diagnostics default on for the generated-loser experiments.
- CSV is written to `dpo_diagnostics.csv` under the training output directory.
- `run_manifest.json` records the manifest adapter parameters through the stage launchers.
- The new D2 manifest adapter does not alter `compute_dpo_loss`; experiment 8 region weighting is intentionally blocked until a wrapper around the existing DPO loss is implemented.

## Evaluation Boundary

Data-only ablations:

- train mask remains full mask;
- loss remains original full-video DPO loss;
- partial masks only generate offline losers;
- eval uses existing VBench and qualitative side-by-side.

Task partial-mask ablation:

- train mask becomes partial mask;
- first version should use `M_train = M_gen`;
- later loss-region studies can compare full, mask-only, and region-weighted losses.

## D2 Manifest Entrypoints

Use `dpo_dataset_type=generated_loser_manifest`.

| Experiment | Manifest | train_mask_mode | mask_from_manifest | loss_region_mode | Status |
| --- | --- | --- | --- | --- | --- |
| 5 | `selected_primary_comp.repaired.jsonl` | `full` | `false` | `full` | dataset ready |
| 6 | `selected_primary_nocomp.repaired.jsonl` | `full` | `false` | `full` | dataset ready |
| 7 | `selected_primary_comp.repaired.jsonl` | `partial` | `true` | `full` | dataset ready |
| 8 | `selected_primary_comp.repaired.jsonl` | `partial` | `true` | `region` | dataset ready, loss wrapper pending |
