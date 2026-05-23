# official_videodpo_diffueraser_data_fullmask_loser

Purpose: data-only ablation for official-VideoDPO DiffuEraser.

## Definition

Changed:

- Loser videos are generated offline by video inpainting models using a full mask.
- Supports DiffuEraser, ProPainter, CoCoCo, and MiniMax-Remover once each runtime is confirmed.

Not changed:

- Win remains the original VideoDPO winner.
- Training remains the official VideoDPO / DiffuEraser full-mask bridge.
- Partial masks are not passed to training.
- DPO loss, dataset semantics, and metric semantics are not changed.

This is a **data-only ablation**.

## Data Contract

| Field | Value |
| --- | --- |
| win | VideoDPO winner |
| raw_loser | `video_inpainting_model(win, full_mask)` |
| final_loser | `raw_loser` |
| mask for loser generation | full |
| mask for training | full |
| comp | false / not meaningful |
| loser generation | offline |
| changed variable | data only |

## Mask Convention

The experiment semantics are full-frame / full-video inpainting loser generation. The actual mask pixel value must be audited per generator model.

For the current DiffuEraser bridge, local code confirms:

- `training/dpo/dataset/videodpo_fullmask_dataset.py`: training full-hole bridge uses a black conditioning image, and internal BrushNet mask value `0.0` means unknown/hole.
- `tools/generate_diffueraser_fullmask_vbench.py`: internal `0.0` maps to PIL pixel `255`, because DiffuEraser PIL white is converted to internal `0/hole`.

Do not apply this convention blindly to ProPainter, CoCoCo, or MiniMax-Remover; audit each model before generation.

## PAI Audit Requirement

Before running, verify for each loser generation model:

- code path;
- conda/env path;
- weights;
- generation script;
- README/runbook;
- mask convention.

If a model is not found, record `未找到`; if its runnable state is ambiguous, record `未确认`.

## Metrics / Diagnostics

Evaluate PSNR, SSIM, VBench, and qualitative SBS. During training, monitor `implicit_acc`, `win_gap`, `lose_gap`, `mse_w`, `ref_mse_w`, `mse_l`, `ref_mse_l`, `loser_dominant_ratio`, `sigma_term`, and `grad_norm`.

## Entry Points

- `scripts/run_generate_losers.sh`: dry-run wrapper around `tools.offline_loser_generation`.
- Training should reuse `official_videodpo_diffueraser` launchers with this experiment's generated manifest/data root.
