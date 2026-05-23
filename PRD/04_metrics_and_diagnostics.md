# Metrics And Diagnostics

## Metrics

Use:

- PSNR
- SSIM
- VBench full score and sub-dimensions
- Qualitative SBS videos

For DiffuEraser evaluation, prefer frame-wise metric transfer rather than mp4 transfer when measuring image quality.

## DPO Diagnostics

Always monitor more than `dpo_loss` and `implicit_acc`.

Required diagnostics:

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
- PSNR
- SSIM
- VBench

Interpretation notes:

- `implicit_acc=1` with `dpo_loss=0` can be a shortcut signal, not a quality signal.
- A high `loser_dominant_ratio` means the model may be winning preference mostly by degrading loser-side behavior.
- Check whether winner quality is preserved using PSNR/SSIM and final sampled videos.

## Loss Supervision Region: Future Study

Do not change loss in this refactor. Record these options for later:

- Full video loss.
- Mask-only loss.
- Non-mask loss.
- Boundary-weighted loss.
- Separate weights for mask / boundary / outside regions.

## Comp vs No-Comp Gradient Notes

If comp is enabled:

- Win and loser are exactly identical outside the partial mask.
- Preference difference concentrates in the mask region.
- In full-mask bridge training, the model does not know the partial mask, so mask-only loss is not directly valid unless the task changes.

If no-comp is enabled:

- Raw generated loser may differ outside mask due to color/texture/brightness/time drift.
- This can produce preference shortcuts unrelated to local artifact quality.

## Regularization

Do not enable new regularization by default in the scaffold. Decide later based on diagnostics, especially `win_gap`, `lose_gap`, `loser_dominant_ratio`, PSNR, SSIM, and VBench.

## Data-Only vs Task Boundary

For `official_videodpo_diffueraser_data_fullmask_loser`, `official_videodpo_diffueraser_data_partialmask_loser_comp`, and `official_videodpo_diffueraser_data_partialmask_loser_nocomp`, loss should remain the current full-mask bridge loss. The model is not told about the partial mask, so do not add mask-only or boundary-weighted loss there.

For `official_videodpo_diffueraser_task_partialmask`, partial mask enters training. Only in that task-ablation branch does it become meaningful to study mask-only, boundary-weighted, or region-weighted losses.
