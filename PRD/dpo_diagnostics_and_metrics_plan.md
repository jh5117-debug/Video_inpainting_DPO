# DPO Diagnostics And Metrics Plan

This is a plan for later training runs. The current asset-preparation phase
does not start training and does not modify loss math.

## 2026-05-30 Exp5 beta500 Collapse Interpretation

Old Exp5 beta500 is marked failed/collapsed/diagnostic only. The important
diagnostic lesson is that DPO objective saturation is not visual quality:

- `acc=1`, `dpo=0`, and `loss=0` appeared early.
- Stage2 10000 qualitative VBench outputs collapsed into high-frequency noise
  and color explosion.
- VBench dimensions such as dynamic degree, overall consistency, scene,
  spatial relationship, and object class were weak.

This is interpreted as preference-data / optimization failure rather than task
failure. Exp3 remains evidence that the DiffuEraser DPO task bridge can work.

The immediate rerun keeps the loss math unchanged and changes only the
optimization strength and duration:

```text
beta_dpo = 10
stage1_max_steps = 4000
stage2_max_steps = 4000
sft_reg_weight = 0
validation_steps = 999999
```

Do not change `compute_dpo_loss`, add SFT regularization, or enable Exp8 region
loss in this pass.

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
