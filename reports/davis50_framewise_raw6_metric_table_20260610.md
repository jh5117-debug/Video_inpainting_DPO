# DAVIS-50 Frame-wise raw6 Metric Table

Protocol: DAVIS-50, 24 frames, raw6, no PCM, D+G off, hard comp, frame-wise
metrics from `inference/metrics.py`.

| Method | Weight | Step | D+G | PSNR | SSIM | LPIPS | VFID | TC |
|---|---|---|---|---:|---:|---:|---:|---:|
| DiffuEraser | step48000 | raw6 | off | 32.7018 | 0.9710 | TBD | TBD | TBD |
| Exp9-1 | DPO-S1 + SFT-S2 | raw6 | off | 32.7402 | 0.9719 | TBD | TBD | TBD |
| Exp9-2 | DPO-S1 + DPO-S2 | raw6 | off | 32.8080 | 0.9717 | TBD | TBD | TBD |
| Exp10-1 | DPO-S1 + SFT-S2 | raw6 | off | 32.9083 | 0.9716 | TBD | TBD | TBD |
| Exp10-2 | DPO-S1 + DPO-S2 | raw6 | off | 32.8791 | 0.9715 | TBD | TBD | TBD |
| Exp11-1 | DPO-S1 + SFT-S2 | raw6 | off | 32.8726 | 0.9716 | TBD | TBD | TBD |
| Exp11-2 | DPO-S1 + DPO-S2 | raw6 | off | 32.9034 | 0.9716 | TBD | TBD | TBD |

`TBD` columns require running the same protocol with explicit metric assets:

```bash
COMPUTE_LPIPS=1 \
COMPUTE_VFID=1 I3D_MODEL_PATH=/path/to/i3d_rgb_imagenet.pt \
COMPUTE_TC=1 TC_MODEL_PATH=/path/to/local/open_clip_vit_h_14 \
bash scripts/run_exp09_10_11_davis50_framewise_protocol_pai.sh
```

Do not compute these columns from mp4 outputs if the table is intended to match
the fixed frame-wise protocol.
