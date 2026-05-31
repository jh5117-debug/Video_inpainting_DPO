# DPO Diagnostics And Metrics Plan

This is a plan for later training runs. The current asset-preparation phase
does not start training and does not modify loss math.

## 2026-05-31 Exp5 Collapse Interpretation

Old Exp5 beta500 and unanchored Exp5 beta10 s1s2 4000 are both marked
failed/collapsed/diagnostic only. The important diagnostic lesson is that DPO
objective saturation is not visual quality:

- `acc=1`, low `dpo_loss`, and saturated `sigma_term` appeared early.
- Stage2 qualitative VBench outputs collapsed into high-frequency noise,
  universal stripe textures, and color explosion.
- `mse_w >> ref_mse_w` and `mse_l >> ref_mse_l`: the policy damaged the winner
  and damaged the loser even more.
- `win_gap` and `lose_gap` grew while ranking accuracy remained high.
- VBench dimensions such as dynamic degree, overall consistency, scene,
  spatial relationship, and object class were weak.

This is interpreted as preference-data / optimization failure rather than task
failure. Exp3 remains evidence that the DiffuEraser DPO task bridge can work.

The next rerun changes the objective minimally by anchoring the winner:

```text
beta_dpo = 10
lose_gap_weight = 0.25
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
winner_gap_reg_margin = 0.0
sft_reg_weight = 0.0
stage1_max_steps = 4000
stage2_max_steps = 4000
validation_steps = 999999
```

The old default loss path must remain unchanged when the new weights are 0.
Do not enable Exp8 region loss in this pass.

Winner-anchor objective:

```text
L_total = L_DPO
        + lambda_abs * model_losses_w.mean()
        + lambda_gap * ReLU(model_losses_w - ref_losses_w - margin).mean()
```

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
- `winner_abs_reg`
- `winner_abs_reg_weight`
- `winner_gap_reg`
- `winner_gap_reg_weight`
- `winner_gap_reg_margin`
- `relu_win_gap_mean`
- `relu_win_gap_max`
- `win_gap_positive_ratio`
- `mse_w_over_ref_mse_w`
- `mse_l_over_ref_mse_l`
- `anchored_total_loss`
- `lose_gap_weight`

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

## Exp7 Gate Diagnostics

`exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500` is the first
winner-anchored partial-mask task gate. It keeps the D2 comp data and uses:

```text
train_mask_mode = partial
mask_from_manifest = true
loss_region_mode = full
beta_dpo = 10
lose_gap_weight = 0.25
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
stage1_steps = 1500
stage2_steps = 1500
```

Gate validation is qual30 side-by-side plus DPO diagnostics summary. Full
VBench is disabled by default for the gate.

The gate summary must report mean, median, p90, max, and these fractions for
Stage1 and Stage2:

- `dpo_loss < 1e-3`
- `implicit_acc > 0.99`
- `mse_w_over_ref_mse_w > 5`
- `win_gap > 0.5`
- `sigma_term > 0.99`
- `kl_divergence > 1.0`

The summary writes a coarse verdict: `PASS_LIKELY`, `RISKY`, or `FAIL_LIKELY`.
It is diagnostic only; do not auto-kill based on the verdict.
