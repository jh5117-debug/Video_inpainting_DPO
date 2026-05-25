#!/usr/bin/env bash
set -uo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root" || exit 1

export VIDEO_DPO_TRAIN_DATA_YAML="${VIDEO_DPO_TRAIN_DATA_YAML:-/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml}"
export DIFFUERASER_PYTHON="${DIFFUERASER_PYTHON:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/downloads/sd_inpaint_hf_extract/stable-diffusion-inpainting}"
export VAE_PATH="${VAE_PATH:-/mnt/nas/hj/weights/sd-vae-ft-mse}"
export PCM_WEIGHTS_PATH="${PCM_WEIGHTS_PATH:-/mnt/nas/hj/weights/PCM_Weights}"
export THIRD_PARTY_VIDEO_INPAINTING_ROOT="${THIRD_PARTY_VIDEO_INPAINTING_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser}"

bash scripts/h20_launch_fullmask_losers_diffueraser_sharded.sh "$@"
