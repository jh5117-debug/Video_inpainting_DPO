#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp28_inner_boundary}"
cd "${ROOT}"

PAIR_ID="${1:-${PAIR_ID:-pairA_inner2_cli4}}"
CONTROL_MODEL="${2:-${CONTROL_MODEL:-fresh_control_A}}"
CANDIDATE_MODEL="${3:-${CANDIDATE_MODEL:-inner2_candidate}}"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
PAIR_ROOT="${PAIR_ROOT:-${OUTPUT_ROOT}/experiments/dpo/exp28_fine_inner_boundary_sweep/pairs/${PAIR_ID}}"
EXPORT_ROOT="${EXPORT_ROOT:-${OUTPUT_ROOT}/experiments/dpo/exp28_fine_inner_boundary_sweep/eval_exports/${PAIR_ID}}"
EVAL_ROOT="${EVAL_ROOT:-${OUTPUT_ROOT}/logs/autoresearch/exp28_fine_inner_boundary_sweep/paired_davis50_eval/${PAIR_ID}}"

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
EVAL_GPU="${EVAL_GPU:-1}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${EXPORT_ROOT}" "${EVAL_ROOT}" reports logs/pipelines

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "[exp28-eval][ERROR] missing ${label}: ${path}" >&2
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
      echo "[exp28-eval][ERROR] refusing to archive outside export root: ${real_out}" >&2
      exit 2
    fi
    archive="${out}.incomplete.$(date +%Y%m%d_%H%M%S)"
    mv "${out}" "${archive}"
    echo "[exp28-eval] archived incomplete export ${out} -> ${archive}" >&2
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
    "${PY}" exp28_fine_inner_boundary_sweep/code/export_accelerate_checkpoint_to_diffueraser.py \
      --checkpoint_dir "${ckpt}" \
      --template_weights "${template}" \
      --output_dir "${out}" \
      --model_label "${model}" \
      --stage "${stage}" \
      --step "${step}" \
      --validate_template_keys \
      "${overwrite_args[@]}" \
      > "${out}.export_stdout.log" \
      2> "${out}.export_stderr.log"
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
      --report_path "reports/exp28_${PAIR_ID}_${model}_stage1_${step}_hybrid_key_merge_report.md" \
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
    echo "[exp28-eval] reuse ${label}: ${out}/metrics/summary.csv"
    return 0
  fi
  require_file "${weights}/unet_main/config.json" "${label} unet config"
  require_file "${weights}/brushnet/config.json" "${label} brushnet config"
  local args=(
    --video_root "${VIDEO_ROOT}"
    --mask_root "${MASK_ROOT}"
    --gt_root "${GT_ROOT}"
    --save_path "${out}"
    --label "${label}"
    --diffueraser_path "${weights}"
    --base_model_path "${BASE_MODEL}"
    --vae_path "${VAE}"
    --propainter_model_dir "${PROP}"
    --pcm_weights_path "${PCM}"
    --input_size "${INPUT_SIZE}"
    --video_length "${VIDEO_LENGTH}"
    --num_inference_steps "${NUM_INFERENCE_STEPS}"
    --use_pcm false
    --mask_dilation_iter 0
    --limit_videos "${LIMIT_VIDEOS}"
    --save_videos
    --save_comp_frames
    --compute_lpips
    --device "${DEVICE}"
  )
  if [[ "${COMPUTE_VFID:-0}" == "1" ]]; then
    i3d_path="${I3D_MODEL_PATH:-${ROOT}/weights/i3d_rgb_imagenet.pt}"
    if [[ -f "${i3d_path}" ]]; then
      args+=(--compute_vfid --i3d_model_path "${i3d_path}")
    else
      echo "[exp28-eval][WARN] COMPUTE_VFID=1 but missing I3D model: ${i3d_path}; skipping VFID" >&2
    fi
  fi
  if [[ "${COMPUTE_TC:-0}" == "1" ]]; then
    tc_path="${TC_MODEL_PATH:-${WEIGHTS_DIR}/open_clip_vit_h14}"
    if [[ -e "${tc_path}" ]]; then
      args+=(--compute_tc --tc_model_path "${tc_path}")
    else
      echo "[exp28-eval][WARN] COMPUTE_TC=1 but missing TC model path: ${tc_path}; skipping TC" >&2
    fi
  fi
  if [[ "${COMPUTE_EWARP:-0}" == "1" ]]; then
    raft_path="${RAFT_MODEL_PATH:-${PROP}/raft-things.pth}"
    if [[ -f "${raft_path}" ]]; then
      args+=(--compute_ewarp --raft_model_path "${raft_path}")
    else
      echo "[exp28-eval][WARN] COMPUTE_EWARP=1 but missing RAFT model: ${raft_path}; skipping Ewarp" >&2
    fi
  fi
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PY}" tools/run_davis50_framewise_protocol_eval.py "${args[@]}"
}

for model in "${CONTROL_MODEL}" "${CANDIDATE_MODEL}"; do
  require_file "${PAIR_ROOT}/${model}/stage1/resolved_region_config.json" "${model} stage1 region config"
  require_file "${PAIR_ROOT}/${model}/stage2/resolved_region_config.json" "${model} stage2 region config"
done

run_eval "sft48000_baseline" "${SFT_STAGE2}"

for step in 1000 1500 2000; do
  control_s1="$(export_checkpoint "${CONTROL_MODEL}" stage1 "${step}" "${PAIR_ROOT}/${CONTROL_MODEL}/stage1/last_weights")"
  cand_s1="$(export_checkpoint "${CANDIDATE_MODEL}" stage1 "${step}" "${PAIR_ROOT}/${CANDIDATE_MODEL}/stage1/last_weights")"
  control_s2="$(export_checkpoint "${CONTROL_MODEL}" stage2 "${step}" "${PAIR_ROOT}/${CONTROL_MODEL}/stage2/last_weights")"
  cand_s2="$(export_checkpoint "${CANDIDATE_MODEL}" stage2 "${step}" "${PAIR_ROOT}/${CANDIDATE_MODEL}/stage2/last_weights")"

  control_hybrid="$(build_hybrid "${CONTROL_MODEL}" "${step}" "${control_s1}")"
  cand_hybrid="$(build_hybrid "${CANDIDATE_MODEL}" "${step}" "${cand_s1}")"

  run_eval "fresh_s2_${step}" "${control_s2}"
  run_eval "candidate_s2_${step}" "${cand_s2}"
  run_eval "fresh_stage1_${step}_sft_s2" "${control_hybrid}"
  run_eval "candidate_stage1_${step}_sft_s2" "${cand_hybrid}"
done

"${PY}" exp28_fine_inner_boundary_sweep/code/summarize_exp28_pair_eval.py \
  --pair_id "${PAIR_ID}" \
  --eval_root "${EVAL_ROOT}" \
  --pair_root "${PAIR_ROOT}" \
  --report_prefix "reports/exp28_${PAIR_ID}"
