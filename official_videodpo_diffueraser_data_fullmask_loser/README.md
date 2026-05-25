# official_videodpo_diffueraser_data_fullmask_loser

Purpose: data-only ablation for official-VideoDPO DiffuEraser.

## Definition

Changed:

- Loser videos are generated offline by DiffuEraser using a full mask.
- Current省时版 manifest/report must record `generation_source = diffueraser_only`.

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
| candidates | one full-mask DiffuEraser candidate per winner |
| raw_loser | `video_inpainting_model(win, full_mask)` |
| final_loser | selected primary `raw_loser` |
| mask for loser generation | full |
| mask for training | full |
| comp | false / not meaningful |
| loser generation | offline |
| generation_source | `diffueraser_only` |
| process_name | `lingbot-world` |
| changed variable | data only |

Selection uses `medium_hard_balanced_selection_v1` and writes:

- `manifests/candidates_all.jsonl`
- `manifests/selected_primary_fullmask.jsonl`
- `manifests/selected_secondary_fullmask.jsonl`

First-version DPO training reads only `selected_primary_fullmask.jsonl`.

## Mask Convention

The experiment semantics are full-frame / full-video inpainting loser generation. The actual mask pixel value must be audited per generator model.

For the current DiffuEraser bridge, local code confirms:

- `training/dpo/dataset/videodpo_fullmask_dataset.py`: training full-hole bridge uses a black conditioning image, and internal BrushNet mask value `0.0` means unknown/hole.
- `tools/generate_diffueraser_fullmask_vbench.py`: internal `0.0` maps to PIL pixel `255`, because DiffuEraser PIL white is converted to internal `0/hole`.

Do not apply this convention blindly to ProPainter, CoCoCo, or MiniMax-Remover; audit each model before generation.

## H20 / PAI Audit Requirement

Before running, verify the DiffuEraser generation stack:

- code path;
- conda/env path;
- weights;
- generation script;
- README/runbook;
- mask convention.

If a required asset is not found, record `未找到`; if its runnable state is ambiguous, record `未确认`.

## Metrics / Diagnostics

Evaluate PSNR, SSIM, VBench, and qualitative SBS. During training, monitor `implicit_acc`, `win_gap`, `lose_gap`, `mse_w`, `ref_mse_w`, `mse_l`, `ref_mse_l`, `loser_dominant_ratio`, `sigma_term`, and `grad_norm`.

## Entry Points

- `scripts/run_generate_losers.sh`: dry-run wrapper around `tools.offline_loser_generation`.
- `scripts/h20_audit_fullmask_generation_readiness.sh`: H20 readiness audit.
- `scripts/h20_launch_fullmask_losers_diffueraser_sharded.sh`: real H20 D1 fullmask DiffuEraser-only sharded generation.
- `scripts/pai_launch_fullmask_losers_diffueraser_sharded.sh`: PAI wrapper for the same D1 launcher.
- `tools/inspect_generated_loser_manifest_videos.py`: check manifest media paths, 16-frame count, and 320x512 resolution before full generation.
- `tools/rewrite_generated_loser_manifest_paths.py`: rewrite H20 absolute paths for PAI after data transfer.
- Training should reuse `official_videodpo_diffueraser` launchers with this experiment's generated manifest/data root.
