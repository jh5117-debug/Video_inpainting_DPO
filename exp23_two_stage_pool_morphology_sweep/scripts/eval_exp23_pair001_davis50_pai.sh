#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp23_pool_sweep}"
cd "${ROOT}"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
PAIR_ID="${PAIR_ID:-phaseA_scale1_pair001_outer2_corrected_outer_control_seed20260619_gpus2456}"
PAIR_ROOT="${PAIR_ROOT:-${OUTPUT_ROOT}/experiments/dpo/exp23_two_stage_pool_morphology_sweep/pairs/${PAIR_ID}}"
EXPORT_ROOT="${EXPORT_ROOT:-${OUTPUT_ROOT}/experiments/dpo/exp23_two_stage_pool_morphology_sweep/eval_exports/${PAIR_ID}}"
EVAL_ROOT="${EVAL_ROOT:-${OUTPUT_ROOT}/logs/target_eval/exp23_two_stage_pool_morphology_sweep/${PAIR_ID}}"

DAVIS_ROOT="${DAVIS_ROOT:-/mnt/workspace/hj/nas_hj/data/external/davis_432_240}"
VIDEO_ROOT="${VIDEO_ROOT:-${DAVIS_ROOT}/JPEGImages_432_240}"
MASK_ROOT="${MASK_ROOT:-${DAVIS_ROOT}/test_masks}"
GT_ROOT="${GT_ROOT:-${DAVIS_ROOT}/JPEGImages_432_240}"

WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/nas/hj/weights}"
BASE_MODEL="${BASE_MODEL:-${WEIGHTS_DIR}/stable-diffusion-v1-5}"
VAE="${VAE:-${WEIGHTS_DIR}/sd-vae-ft-mse}"
PROP="${PROP:-/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter}"
PCM="${PCM:-${WEIGHTS_DIR}/PCM_Weights}"
SFT_STAGE2="${SFT_STAGE2:-/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000}"

INPUT_SIZE="${INPUT_SIZE:-432x240}"
VIDEO_LENGTH="${VIDEO_LENGTH:-24}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-6}"
LIMIT_VIDEOS="${LIMIT_VIDEOS:-50}"
EVAL_GPU="${EVAL_GPU:-2}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${EXPORT_ROOT}" "${EVAL_ROOT}" reports logs/pipelines

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "[exp23-eval][ERROR] missing ${label}: ${path}" >&2
    exit 2
  fi
}

is_export_ready() {
  local out="$1"
  [[ -f "${out}/export_manifest.json" ]] &&
    [[ -f "${out}/unet_main/config.json" ]] &&
    [[ -f "${out}/unet_main/diffusion_pytorch_model.safetensors" ]] &&
    [[ -f "${out}/brushnet/config.json" ]] &&
    [[ -f "${out}/brushnet/diffusion_pytorch_model.safetensors" ]]
}

is_hybrid_ready() {
  local out="$1"
  [[ -f "${out}/hybrid_manifest.json" ]] &&
    [[ -f "${out}/last_weights/unet_main/config.json" ]] &&
    [[ -f "${out}/last_weights/unet_main/diffusion_pytorch_model.safetensors" ]] &&
    [[ -f "${out}/last_weights/brushnet/config.json" ]] &&
    [[ -f "${out}/last_weights/brushnet/diffusion_pytorch_model.safetensors" ]]
}

archive_incomplete_export_dir() {
  local out="$1"
  if [[ -e "${out}" ]]; then
    local real_out real_root archive
    real_out="$(realpath -m "${out}")"
    real_root="$(realpath -m "${EXPORT_ROOT}")"
    if [[ "${real_out}" != "${real_root}"/* ]]; then
      echo "[exp23-eval][ERROR] refusing to archive outside export root: ${real_out}" >&2
      exit 2
    fi
    archive="${out}.incomplete.$(date +%Y%m%d_%H%M%S)"
    mv "${out}" "${archive}"
    echo "[exp23-eval] archived incomplete export ${out} -> ${archive}" >&2
  fi
}

export_checkpoint() {
  local model="$1"
  local stage="$2"
  local step="$3"
  local template="$4"
  local ckpt="${PAIR_ROOT}/${model}/${stage}/checkpoint-${step}"
  local out="${EXPORT_ROOT}/${model}_${stage}_${step}_weights"
  require_file "${ckpt}/model.safetensors" "${model} ${stage} checkpoint-${step} model"
  require_file "${ckpt}/model_1.safetensors" "${model} ${stage} checkpoint-${step} model_1"
  if ! is_export_ready "${out}"; then
    local overwrite_args=()
    if [[ -e "${out}" ]]; then
      overwrite_args=(--overwrite)
    fi
    local export_log="${EXPORT_ROOT}/${model}_${stage}_${step}_export_stdout.log"
    local export_err="${EXPORT_ROOT}/${model}_${stage}_${step}_export_stderr.log"
    "${PY}" exp23_two_stage_pool_morphology_sweep/code/export_accelerate_checkpoint_to_diffueraser.py \
      --checkpoint_dir "${ckpt}" \
      --template_weights "${template}" \
      --output_dir "${out}" \
      --model_label "${model}" \
      --stage "${stage}" \
      --step "${step}" \
      --validate_template_keys \
      "${overwrite_args[@]}" \
      > "${export_log}" \
      2> "${export_err}"
  fi
  printf '%s\n' "${out}"
}

build_hybrid() {
  local model="$1"
  local step="$2"
  local dpo_stage1_weights="$3"
  local out="${EXPORT_ROOT}/${model}_stage1_${step}_hybrid_sft_s2"
  if ! is_hybrid_ready "${out}"; then
    if [[ -e "${out}" ]]; then
      archive_incomplete_export_dir "${out}"
    fi
    mkdir -p "${out}"
    "${PY}" tools/build_diffueraser_dpoS1_sftS2_hybrid.py \
      --dpo_stage1_weights "${dpo_stage1_weights}" \
      --sft_stage2_weights "${SFT_STAGE2}" \
      --output_dir "${out}" \
      --strict false \
      --report_path "reports/exp23_${PAIR_ID}_${model}_stage1_${step}_hybrid_key_merge_report.md" \
      > "${out}/hybrid_builder_stdout.log" \
      2> "${out}/hybrid_builder_stderr.log"
  fi
  printf '%s\n' "${out}/last_weights"
}

run_eval() {
  local label="$1"
  local weights="$2"
  local out="${EVAL_ROOT}/${label}"
  if [[ -f "${out}/metrics/summary.csv" && "${FORCE_EVAL:-0}" != "1" ]]; then
    echo "[exp23-eval] reuse ${label}: ${out}/metrics/summary.csv"
    return 0
  fi
  require_file "${weights}/unet_main/config.json" "${label} unet config"
  require_file "${weights}/brushnet/config.json" "${label} brushnet config"
  local args=(
    --video_root "${VIDEO_ROOT}" \
    --mask_root "${MASK_ROOT}" \
    --gt_root "${GT_ROOT}" \
    --save_path "${out}" \
    --label "${label}" \
    --diffueraser_path "${weights}" \
    --base_model_path "${BASE_MODEL}" \
    --vae_path "${VAE}" \
    --propainter_model_dir "${PROP}" \
    --pcm_weights_path "${PCM}" \
    --input_size "${INPUT_SIZE}" \
    --video_length "${VIDEO_LENGTH}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --use_pcm false \
    --mask_dilation_iter 0 \
    --limit_videos "${LIMIT_VIDEOS}" \
    --save_videos \
    --save_comp_frames \
    --compute_lpips \
    --device "${DEVICE}"
  )
  if [[ "${COMPUTE_VFID:-0}" == "1" ]]; then
    args+=(--compute_vfid --i3d_model_path "${I3D_MODEL_PATH:-${ROOT}/weights/i3d_rgb_imagenet.pt}")
  fi
  if [[ "${COMPUTE_TC:-0}" == "1" ]]; then
    args+=(--compute_tc --tc_model_path "${TC_MODEL_PATH:-${WEIGHTS_DIR}/open_clip_vit_h14}")
  fi
  if [[ "${COMPUTE_EWARP:-0}" == "1" ]]; then
    args+=(--compute_ewarp --raft_model_path "${RAFT_MODEL_PATH:-${PROP}/raft-things.pth}")
  fi
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PY}" tools/run_davis50_framewise_protocol_eval.py "${args[@]}"
}

for model in fresh_exp11_outer_b075 candidate_scale1_outer2_b075; do
  require_file "${PAIR_ROOT}/${model}/stage1/resolved_region_config.json" "${model} stage1 region config"
  require_file "${PAIR_ROOT}/${model}/stage2/resolved_region_config.json" "${model} stage2 region config"
done

run_eval "sft48000_baseline" "${SFT_STAGE2}"

for step in 1000 1500 2000; do
  fresh_s1="$(export_checkpoint fresh_exp11_outer_b075 stage1 "${step}" "${PAIR_ROOT}/fresh_exp11_outer_b075/stage1/last_weights")"
  cand_s1="$(export_checkpoint candidate_scale1_outer2_b075 stage1 "${step}" "${PAIR_ROOT}/candidate_scale1_outer2_b075/stage1/last_weights")"
  fresh_s2="$(export_checkpoint fresh_exp11_outer_b075 stage2 "${step}" "${PAIR_ROOT}/fresh_exp11_outer_b075/stage2/last_weights")"
  cand_s2="$(export_checkpoint candidate_scale1_outer2_b075 stage2 "${step}" "${PAIR_ROOT}/candidate_scale1_outer2_b075/stage2/last_weights")"

  fresh_hybrid="$(build_hybrid fresh_exp11_outer_b075 "${step}" "${fresh_s1}")"
  cand_hybrid="$(build_hybrid candidate_scale1_outer2_b075 "${step}" "${cand_s1}")"

  run_eval "fresh_s2_${step}" "${fresh_s2}"
  run_eval "candidate_s2_${step}" "${cand_s2}"
  run_eval "fresh_stage1_${step}_sft_s2" "${fresh_hybrid}"
  run_eval "candidate_stage1_${step}_sft_s2" "${cand_hybrid}"
done

"${PY}" exp23_two_stage_pool_morphology_sweep/code/summarize_exp23_pair_eval.py \
  --pair_id "${PAIR_ID}" \
  --eval_root "${EVAL_ROOT}" \
  --pair_root "${PAIR_ROOT}" \
  --report_prefix reports/exp23_pair001
