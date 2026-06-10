# SFT48000 DAVIS-50 Metric Protocol

Status: fixed and verified for current Exp9/10/11 comparison.

## Protocol

- Dataset: DAVIS-50 under `/mnt/workspace/hj/nas_hj/data/external/davis_432_240`
- Frames: 24 frames per video for validation
- Weight: `/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000`
- Step: raw6, `NUM_INFERENCE_STEPS=6`
- PCM: off
- D+G: off (`MASK_DILATION_ITER=0`, no Gaussian blur during comp)
- Composition: hard comp outside mask
- Metric path: frame-wise in-memory hard-composited frames
- Metric backend: `inference/metrics.py`
- Do not compute PSNR/SSIM/LPIPS/VFID/TC from `diffueraser_comp.mp4` when filling the canonical table.

## Tooling

```bash
tools/run_davis50_framewise_protocol_eval.py
scripts/run_exp09_10_11_davis50_framewise_protocol_pai.sh
```

The wrapper supports optional metric assets explicitly:

```bash
COMPUTE_LPIPS=1 \
COMPUTE_VFID=1 I3D_MODEL_PATH=/path/to/i3d_rgb_imagenet.pt \
COMPUTE_TC=1 TC_MODEL_PATH=/path/to/local/open_clip_vit_h14 \
SAVE_VIDEOS=0 \
bash scripts/run_exp09_10_11_davis50_framewise_protocol_pai.sh
```

`SAVE_VIDEOS=0` keeps metric-only reruns from duplicating visual outputs.

## Verified Runs

PSNR-only protocol recovery run:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp09_10_11_20260610_framewise_raw6_davis50_v2
```

Final all-metric pass:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp09_10_11_20260611_050013_framewise_raw6_davis50_allmetrics
```

SFT48000 all-metric baseline:

```text
PSNR = 32.6755
SSIM = 0.9702
LPIPS = 0.0168
VFID = 0.1940
TC = 0.9708
rows = 50
```

This remains close to the expected near-32 raw6 SFT48000 regime and confirms that the high score requires frame-wise hard-comp metrics rather than mp4 round-trip metrics.
