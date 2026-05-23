# diffueraser_reproduction_sft

Purpose: organize the completed DiffuEraser reproduction, metric setting search, frame-wise metric fixes, and YouTube-VOS SFT work.

## Scope

Changed:

- DiffuEraser reproduction and inference/evaluation wrappers.
- Metric setting search and quality checks.
- YouTube-VOS SFT entrypoints and notes.

Not changed by this scaffold:

- DiffuEraser model architecture.
- Existing DPO loss.
- Existing dataset semantics.
- Existing metric semantics.

## Current Best Evaluation Setting

- Denoise steps: `6`
- PCM acceleration: off
- Final mask dilation / compositing Gaussian blur: off
- Metric transport: frame-wise transfer instead of mp4 transport, because mp4 compression can reduce image quality.

## Existing Code To Reuse

- `diffueraser/`
- `train_DiffuEraser_stage1.py`
- `train_DiffuEraser_stage2.py`
- `tools/generate_diffueraser_fullmask_vbench.py`
- `DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh`
- `training/sft/`

## Metrics

Track PSNR, SSIM, VBench, and qualitative SBS outputs. For DPO-related follow-up, also track the diagnostics documented in `PRD/04_metrics_and_diagnostics.md`.
