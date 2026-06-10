# SFT48000 DAVIS-50 Metric Protocol

Status: fixed for current Exp9/10/11 comparison.

## Protocol

This is the validation protocol that reproduces the high SFT-48000 DAVIS score
near/above 32 PSNR.

- Dataset: DAVIS-50 under `/mnt/workspace/hj/nas_hj/data/external/davis_432_240`
- Frames: 24 frames per video for validation
- Weight: `/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000`
- Step: raw6, `NUM_INFERENCE_STEPS=6`
- PCM: off
- D+G: off
  - mask dilation: 0
  - Gaussian blur during comp: off
- Composition: hard comp outside mask
- Metric path: frame-wise in-memory hard-composited frames
- Do not compute PSNR/SSIM from `diffueraser_comp.mp4`; mp4 round-trip lowers PSNR.
- Metric backend: `inference/metrics.py`

The current reproducible tool is:

```bash
tools/run_davis50_framewise_protocol_eval.py
```

The PAI wrapper for SFT48000 and Exp9/10/11 is:

```bash
scripts/run_exp09_10_11_davis50_framewise_protocol_pai.sh
```

## Metrics

Default metrics:

- PSNR
- SSIM
- mask-region PSNR
- mask-region SSIM
- outside-region diff sanity check

Optional metrics, enabled only when local assets are explicitly available:

- LPIPS: `--compute_lpips`
- VFID: `--compute_vfid --i3d_model_path <i3d_rgb_imagenet.pt>`
- TC: `--compute_tc --tc_model_path <local open_clip model dir>`
- Ewarp diagnostic: `--compute_ewarp --raft_model_path <raft-things.pth>`

Optional metrics must not silently download assets during a production run.
If the asset path is missing, the run should fail before launching.

## Verified Run

Verified on PAI with:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp09_10_11_20260610_framewise_raw6_davis50_v2
```

SFT48000 baseline:

```text
rows=50
whole_video_psnr_mean=32.701787750340394
whole_video_ssim_mean=0.9709981812091598
mask_region_psnr_mean=23.85532082067563
mask_region_ssim_mean=0.8011341467190402
outside_region_diff_mean_mean=0.0
```

This confirms the frame-wise protocol and explains why older mp4-based metric
runs reported values around 30 PSNR.
