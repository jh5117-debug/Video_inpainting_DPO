# VideoPainter Adapter Gate2000 DAVIS Eval

## Protocol

- Baseline: VideoPainter official branch checkpoint.
- Adapter: Exp14 gate2000 `last_weights`.
- Dataset: DAVIS under `/mnt/workspace/hj/nas_hj/data/external/davis_432_240`.
- Output: `/mnt/nas/hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/logs/target_eval/exp14_videopainter_adapter_gate2000_davis`.
- VideoPainter inference steps: 50.
- VideoPainter frames per clip cap: 49; clips are trimmed to 4k+1 frames.
- Hard comp: prediction inside mask + GT outside mask.
- Mask dilation: off.
- Gaussian blur: off.
- VBench: not used.
- Metric backend: `inference/metrics.py`.

## Summary

```json
[
  {
    "method": "VideoPainter adapter",
    "model": "adapter",
    "PSNR": 29.802756917793353,
    "SSIM": 0.9579541300611826,
    "strict_mask_pixel_psnr": 18.1594574149052,
    "LPIPS": NaN,
    "number_of_videos": 50,
    "number_of_frames": 2366,
    "eval_protocol": "DAVIS frame-wise hard-comp, no mask dilation, no Gaussian blur, no VBench"
  },
  {
    "method": "VideoPainter baseline",
    "model": "baseline",
    "PSNR": 31.61239218545971,
    "SSIM": 0.9607946258068816,
    "strict_mask_pixel_psnr": 19.96909268257156,
    "LPIPS": NaN,
    "number_of_videos": 50,
    "number_of_frames": 2366,
    "eval_protocol": "DAVIS frame-wise hard-comp, no mask dilation, no Gaussian blur, no VBench"
  }
]
```

## Success Candidates

- rollerblade: PSNR 17.7613 -> 31.3885, SSIM 0.8747 -> 0.9814
- scooter-black: PSNR 20.4523 -> 29.1181, SSIM 0.9277 -> 0.9815
- dog-agility: PSNR 19.4185 -> 26.2765, SSIM 0.8698 -> 0.9499
- bus: PSNR 33.0869 -> 38.2354, SSIM 0.9956 -> 0.9981
- motorbike: PSNR 18.7211 -> 21.8379, SSIM 0.8943 -> 0.9141
- libby: PSNR 35.3087 -> 37.7320, SSIM 0.9783 -> 0.9890
- bear: PSNR 31.2278 -> 33.6257, SSIM 0.9791 -> 0.9857
- flamingo: PSNR 30.9764 -> 33.1397, SSIM 0.9711 -> 0.9824

## Tie Candidates

- none

## Failure Candidates

- hockey: PSNR 45.7968 -> 32.7162, SSIM 0.9930 -> 0.9594
- paragliding-launch: PSNR 40.1913 -> 28.1909, SSIM 0.9859 -> 0.9379
- hike: PSNR 42.8295 -> 32.5494, SSIM 0.9950 -> 0.9780
- car-turn: PSNR 40.5050 -> 30.4509, SSIM 0.9936 -> 0.9710
- dog: PSNR 36.1422 -> 26.9412, SSIM 0.9834 -> 0.9333
- dance-jump: PSNR 37.7506 -> 29.7400, SSIM 0.9888 -> 0.9712
- bmx-bumps: PSNR 33.1447 -> 25.4419, SSIM 0.9806 -> 0.9417
- swing: PSNR 35.0495 -> 29.3996, SSIM 0.9886 -> 0.9705

## Interpretation

This report should be read together with `reports/videopainter_adapter_gate2000_dpo_diag_summary.md`.
The existing training diagnostics were already flagged as DPO_SATURATED / LOSER_DOMINANT / GRAD_SPIKE_OBSERVED,
so a metric or visual drop would indicate the current DPO objective is not yet stable for VideoPainter.

## Final Decision

The full DAVIS50 result is negative for the adapter:

```text
adapter_psnr_delta = -1.8096
adapter_ssim_delta = -0.0028
adapter_strict_mask_psnr_delta = -1.8096
psnr_improved_videos = 16 / 50
psnr_dropped_videos = 34 / 50
median_psnr_delta = -1.4387
```

The adapter checkpoint is real and was loaded without fallback, but this
gate2000 run should not be continued as a longer VideoPainter adapter training
run. If VideoPainter adaptation is revisited, it needs a redesigned objective or
data pairing rather than more steps of this same saturated loser-dominant setup.
