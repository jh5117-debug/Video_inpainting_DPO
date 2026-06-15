# VideoPainter DPO Trainer Preflight

Date: 2026-06-16 CST
Host: dsw-753014-dc85766cb-4v2jj

## Status

blocked_before_preflight

## What Passed

- Clean Exp14 worktree was created at:
  `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate`
- Exp14 trainer exists and passes `py_compile`.
- Exp14 gate launcher passes `bash -n`.
- VideoPainter code repo was synced from HAL to:
  `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter`
- Required VideoPainter code entries exist:
  - `train/VideoPainter.sh`
  - `train/train_cogvideox_inpainting_i2v_video.py`
  - `infer/inpaint.py`
  - `evaluate/eval_inpainting.py`
- YouTube-VOS train data exists.
- DAVIS eval data exists.
- Generated-loser manifest exists and does not contain `/home/nvme01`.
- GPUs are available.

## Blocker

The required VideoPainter weights are missing on both PAI and HAL:

- `third_party/VideoPainter/ckpt/CogVideoX-5b-I2V`
- `third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch`
- `/mnt/nas/hj/weights/CogVideoX-5b-I2V`
- `/mnt/nas/hj/weights/VideoPainter`
- HAL checked paths under `/home/hj/dpo-2-1-exp` and `/home/hj/weights`

Without these weights, the trainer cannot construct:

- trainable VideoPainter policy branch
- frozen VideoPainter reference branch
- same-checkpoint policy/reference pair

Therefore it cannot compute:

- `m_w`
- `m_l`
- `m_w_ref`
- `m_l_ref`

## Decision

Do not run trainer preflight.
Do not launch gate2000.
Do not run upstream VideoPainter official training as a fallback.
Do not download giant unknown weights automatically.

## Required Next Step

Provide or mount the official VideoPainter / CogVideoX checkpoints, then rerun the gate launcher with:

```bash
VIDEO_PAINTER_BASE_MODEL=/path/to/CogVideoX-5b-I2V \
VIDEO_PAINTER_CHECKPOINT_ROOT=/path/to/VideoPainter/checkpoints/branch \
VIDEO_PAINTER_REFERENCE_CHECKPOINT_ROOT=/path/to/VideoPainter/checkpoints/branch \
bash exp14_adapter_videopainter/scripts/launch_videopainter_adapter_gate2000_pai.sh
```
