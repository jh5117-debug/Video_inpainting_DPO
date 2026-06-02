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

## 2026-06-01 Exp7 Partial-Mask Evaluation Addendum

Exp7 gate1500 full-mask qual30 is now treated as **failed /
task-mismatched**, not as a final task verdict. The run was trained with
`train_mask_mode=partial` and `mask_from_manifest=true`, so the fair gate eval
must use the same D2 manifest winner video and mask.

Required partial-mask eval:

```text
script = scripts/eval_exp7_partialmask_gate.sh
manifest = selected_primary_comp.repaired.jsonl
base = DiffuEraser-base converted_weights_step48000
exp checkpoints =
  Stage1 checkpoint-500 if exported/evaluable
  Stage1 checkpoint-1000 if exported/evaluable
  Stage1 last_weights
  Stage2 last_weights
num_samples = 30
num_samples_metric = 100
seed = 42
```

Side-by-side format:

```text
winner / GT | mask overlay | DiffuEraser-base partial-mask comp | Exp7 partial-mask comp | optional D2 loser comp
```

Metrics:

- `whole_video_psnr`, `whole_video_ssim`
- `mask_region_psnr`, `mask_region_ssim`
- `boundary_psnr`, `boundary_ssim`
- `outside_region_diff_mean`, `outside_region_diff_max`
- `temporal_diff`, `temporal_diff_delta_vs_gt`

The report must answer:

- whether Exp7 beats DiffuEraser-base on the true partial-mask task;
- whether Stage1 early checkpoints beat Stage2 last;
- whether collapse appears mainly in Stage2;
- whether Exp7 full 4000+4000 is worth launching;
- whether to start the prepared no-lose-gap gate.

Do not run full Exp7, full VBench, or Exp8 until this partial-mask report is
reviewed.

## 2026-06-02 Exp7-PM-Gate1500 Result

The partial-mask eval completed successfully after the eval video reader was
changed to ffmpeg rawvideo decoding. The original `imageio` path selected an
incompatible `pyav` backend on PAI and failed with:

```text
AttributeError: 'av.format.ContainerFormat' object has no attribute 'variable_fps'
```

Completed eval:

```text
name = Exp7-PM-Gate1500
output_root = /mnt/nas/hj/H20_Video_inpainting_DPO/logs/partialmask_eval/exp7_gate1500_20260602_000500
side_by_side = 60 mp4
metrics = metrics/summary.csv
report = report.md
dpo_summary = /mnt/nas/hj/H20_Video_inpainting_DPO/reports/exp7_gate1500_dpo_diag_summary.md
```

Metric summary:

| Model | mask_region_psnr_mean | mask_region_ssim_mean | outside_region_diff_mean_mean | temporal_diff_delta_vs_gt_mean |
| --- | ---: | ---: | ---: | ---: |
| DiffuEraser-base | 8.99765 | 0.272146 | 2.91477 | 5.58378 |
| Stage1_last | 9.57079 | 0.288404 | 2.92006 | 12.7824 |
| Stage2_last | 7.88448 | 0.235938 | 2.91600 | 6.52143 |

Diagnostic interpretation:

- Winner-gap control works well enough to prevent the old Exp5 winner damage:
  Stage1/Stage2 `win_gap` stays bounded relative to the unanchored collapse.
- Loser degradation remains the dominant shortcut: Stage2 diagnostics show
  high `mse_l_over_ref_mse_l` and `loser_dominant_ratio` near 1.0 for many
  steps.
- The task-matched metrics indicate Stage1 learned useful partial-mask
  behavior, but Stage2 moved the model in the wrong direction.

Decision:

- Exp7 is not a final success.
- Exp7 is not a total failure either: partial-mask task alignment is validated
  by `Stage1_last` beating DiffuEraser-base on mask-region PSNR/SSIM.
- Do not run full Exp7 4000+4000 or full VBench yet.
- Review the partial-mask side-by-side videos and then prefer the prepared
  no-lose-gap gate or a Stage1-focused variant over another long Stage2.

## 2026-06-02 DPO-S1 + SFT-S2 Hybrid Diagnostic Plan

The Stage1/Stage2 roles change how Exp7 diagnostics should be read:

- Stage1 diagnostics and partial-mask metrics primarily speak to spatial /
  appearance quality.
- Stage2 diagnostics primarily speak to temporal/motion adaptation.
- Exp7 Stage1_last beating DiffuEraser-base does not mean final inference
  should be Stage1-only.
- Exp7 Stage2_last regressing means the current DPO Stage2 objective is
  harmful and should stay stopped.

Next diagnostic target:

```text
candidate = DPO Stage1 spatial weights + frozen SFT Stage2 motion weights
eval = true partial-mask manifest eval
full_vbench = disabled
training = none
```

Hybrid eval metrics should keep the existing partial-mask set:

- `whole_video_psnr`, `whole_video_ssim`
- `mask_region_psnr`, `mask_region_ssim`
- `boundary_psnr`, `boundary_ssim`
- `outside_region_diff_mean`, `outside_region_diff_max`
- `temporal_diff_delta_vs_gt`

The hybrid report must answer:

- whether the checkpoint structure safely separates DPO spatial and SFT motion
  weights;
- whether YouTube-VOS SFT Stage2 was found;
- whether DPO Stage1 spatial/BrushNet weights were preserved;
- whether SFT Stage2 motion/temporal weights were preserved;
- which DPO Stage1 checkpoint + SFT Stage2 candidate is best;
- whether the hybrid beats DiffuEraser-base;
- whether the hybrid beats Exp7 DPO Stage1 + DPO Stage2;
- whether DPO Stage2 should remain stopped.

Prepared but not launched:

```text
scripts/launch_exp7_pm_stage1only_ckptsweep_pai.sh
```

This future script is only for producing better DPO Stage1 checkpoints
(`3000` steps, checkpoint every `500`) if the hybrid audit says the current
Stage1 candidates are insufficient. It must not launch Stage2 or full VBench.
## 2026-06-02 Target-Domain Evaluation Priority

Diagnostics must now distinguish bridge-domain health from final target-domain
quality.

Bridge domain:

- VideoDPO
- Purpose: integration, DPO loss behavior, generated-loser manifests,
  partial-mask plumbing, Stage1/Stage2 loading, ablation.
- Metrics: useful for debugging but not final success.

Target domains:

- YouTube-VOS
- DAVIS
- Purpose: final quality decision and demo/reporting.

Current diagnostic interpretation:

- Exp7 full-mask and partial-mask VideoDPO evaluations are diagnostic only.
- DPO Stage2 remains risky because it regresses temporal/motion behavior.
- VideoDPO SFT warmup is not current plan.
- D3 YouTube-VOS generated-loser data is target-domain data preparation, not a
  training trigger by itself.

Required target-domain eval metrics:

- whole-video PSNR / SSIM
- mask-region PSNR / SSIM
- boundary PSNR / SSIM
- outside-region mean/max diff
- temporal flicker or temporal-diff when available
- qualitative side-by-side on YouTube-VOS and DAVIS

Required target-domain eval settings:

- denoise steps = 6
- no PCM
- no Gaussian blur
- no unnecessary mask dilation
- hard comp outside mask
- frame-wise metric path

If a script cannot guarantee these settings, it should write a preflight report
and stop instead of running a mismatched eval.
