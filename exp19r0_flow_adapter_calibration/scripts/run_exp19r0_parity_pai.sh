#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
PYTHON="${PYTHON:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

cd "${REPO_ROOT}"
mkdir -p reports logs/pipelines

"${PYTHON}" exp19r0_flow_adapter_calibration/code/run_exp19r0_parity.py \
  --video_root /mnt/workspace/hj/nas_hj/data/external/davis_432_240/JPEGImages_432_240 \
  --mask_root /mnt/workspace/hj/nas_hj/data/external/davis_432_240/test_masks \
  --gt_root /mnt/workspace/hj/nas_hj/data/external/davis_432_240/JPEGImages_432_240 \
  --save_path /mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp19r0_flow_adapter_calibration/parity \
  --flow_cache_root /mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp19_davis10_propainter_completed_flow \
  --base_model_path /mnt/nas/hj/weights/stable-diffusion-v1-5 \
  --vae_path /mnt/nas/hj/weights/sd-vae-ft-mse \
  --exp11_weights /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/last_weights \
  --exp19_adapter /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/exp19b_boundary_flow_adapter_s2_500_limit100/last_weights/flow_adapter.pt \
  --propainter_model_dir /mnt/nas/hj/data/third_party_video_inpainting/weights/propainter \
  --pcm_weights_path /mnt/nas/hj/weights/PCM_Weights \
  --input_size 432x240 \
  --video_length "${VIDEO_LENGTH:-24}" \
  --nframes "${NFRAMES:-22}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS:-6}" \
  --limit_videos "${LIMIT_VIDEOS:-1}" \
  --seed "${SEED:-1234}" \
  --device cuda
