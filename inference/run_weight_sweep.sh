#!/usr/bin/env bash
set -uo pipefail

# Weight-sweep evaluation runner.
#
# Runs the same OR/BR visualisation + metric pipeline across multiple converted
# DiffuEraser checkpoints. Large assets are intentionally referenced from
# ignored project directories: data/, weights/, and experiments/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

if [[ "${AUTO_CONDA:-1}" == "1" ]] && command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-diffueraser}"
fi

EXP_ROOT="${EXP_ROOT:-${PROJECT_ROOT}/experiments/evaluation/weight_sweep}"
LOG_DIR="${LOG_DIR:-${EXP_ROOT}/logs_${TIMESTAMP}}"
mkdir -p "${LOG_DIR}"

BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/weights/stable-diffusion-v1-5}"
VAE="${VAE:-${PROJECT_ROOT}/weights/sd-vae-ft-mse}"
PROPAINTER="${PROPAINTER:-${PROJECT_ROOT}/weights/propainter}"
PCM="${PCM:-${PROJECT_ROOT}/weights/PCM_Weights}"
I3D="${I3D:-${PROJECT_ROOT}/weights/i3d_rgb_imagenet.pt}"
RAFT="${RAFT:-${PROJECT_ROOT}/weights/propainter/raft-things.pth}"

DIFFUERASER_ORIG="${DIFFUERASER_ORIG:-${PROJECT_ROOT}/weights/diffuEraser}"
DIFFUERASER_FT_8K="${DIFFUERASER_FT_8K:-${PROJECT_ROOT}/weights/diffuEraser/converted_weights_step8000}"
DIFFUERASER_FT_26K="${DIFFUERASER_FT_26K:-${PROJECT_ROOT}/weights/diffuEraser/converted_weights_step26000}"
DIFFUERASER_FT_34K="${DIFFUERASER_FT_34K:-${PROJECT_ROOT}/weights/diffuEraser/converted_weights_step34000}"
DIFFUERASER_FT_48K="${DIFFUERASER_FT_48K:-${PROJECT_ROOT}/weights/diffuEraser/converted_weights_step48000}"

OR_VIDEO="${OR_VIDEO:-${PROJECT_ROOT}/data/external/davis_2017_full_resolution/DAVIS/JPEGImages/Full-Resolution}"
OR_MASK="${OR_MASK:-${PROJECT_ROOT}/data/external/davis_2017_full_resolution/DAVIS/Annotations/Full-Resolution}"
OR_CAPTION="${OR_CAPTION:-${PROJECT_ROOT}/inference/prompt_cache/all_captions_OR.yaml}"

BR_VIDEO="${BR_VIDEO:-${PROJECT_ROOT}/data/external/davis_432_240/JPEGImages_432_240}"
BR_MASK="${BR_MASK:-${PROJECT_ROOT}/data/external/davis_432_240/test_masks}"
BR_GT="${BR_GT:-${PROJECT_ROOT}/data/external/davis_432_240/JPEGImages_432_240}"
BR_CAPTION="${BR_CAPTION:-${PROJECT_ROOT}/inference/prompt_cache/all_captions_BR.yaml}"

INFERENCE_DIR="${PROJECT_ROOT}/inference"
MAX_VIDEOS="${MAX_VIDEOS:-50}"
MIN_FREE_MB="${MIN_FREE_MB:-18000}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
MAX_RETRIES="${MAX_RETRIES:-2}"
LOCK_PREFIX="${LOCK_PREFIX:-/tmp/h20_weight_sweep_gpu}"
OR_ENABLE_VBENCH="${OR_ENABLE_VBENCH:-0}"

read -r -a ALL_GPUS <<< "${GPUS:-0 1 2 3 5 6}"

cleanup() {
    rm -f "${LOCK_PREFIX}"_*.lock
}

trap 'echo ""; echo "[!] Stop requested, killing child processes..."; kill $(jobs -p) 2>/dev/null; wait 2>/dev/null; cleanup; echo "[!] Stopped."; exit 1' INT TERM

acquire_gpu() {
    while true; do
        for gpu_id in "${ALL_GPUS[@]}"; do
            local lockfile="${LOCK_PREFIX}_${gpu_id}.lock"
            if ( set -o noclobber; echo "$$" > "$lockfile" ) 2>/dev/null; then
                local free_mb
                free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d '[:space:]')"
                if [[ -n "$free_mb" ]] && (( free_mb >= MIN_FREE_MB )); then
                    echo "$gpu_id"
                    return 0
                fi
                rm -f "$lockfile"
            fi
        done
        echo "[WAIT] No GPU with >=${MIN_FREE_MB}MB free. Polling in ${POLL_INTERVAL}s..." >&2
        sleep "$POLL_INTERVAL"
    done
}

release_gpu() {
    local gpu_id=$1
    rm -f "${LOCK_PREFIX}_${gpu_id}.lock"
}

require_dir() {
    local label=$1
    local path=$2
    if [[ ! -d "$path" ]]; then
        echo "[MISSING] ${label}: ${path}" >&2
        return 1
    fi
    return 0
}

run_experiment() {
    local gpu=$1 exp_type=$2 ckpt=$3 gs=$4 blend=$5 dil=$6 outdir=$7 logname=$8 diffueraser_path=$9

    local blend_flag=""
    [[ "$blend" == "true" ]] && blend_flag="--blended"

    local vid_root mask_root gt_root caption_yaml extra_args eval_flag
    if [[ "$exp_type" == "OR" ]]; then
        vid_root="$OR_VIDEO"
        mask_root="$OR_MASK"
        gt_root=""
        caption_yaml="$OR_CAPTION"
        extra_args="--height 360 --width 640 --video_length 60 --ref_stride 6"
        eval_flag=""
        [[ "$OR_ENABLE_VBENCH" == "1" ]] && eval_flag="--eval"
    else
        vid_root="$BR_VIDEO"
        mask_root="$BR_MASK"
        gt_root="$BR_GT"
        caption_yaml="$BR_CAPTION"
        extra_args="--video_length 100 --ref_stride 3"
        eval_flag="--eval --no_vbench"
    fi

    local log_path="${LOG_DIR}/${logname}.log"
    echo "[START] ${outdir} on GPU ${gpu} ($(date))"

    {
        echo "[CONFIG] exp_type=${exp_type} ckpt=${ckpt} gs=${gs} blend=${blend} dil=${dil}"
        echo "[CONFIG] diffueraser_path=${diffueraser_path}"
        echo "[CONFIG] video=${vid_root}"
        echo "[CONFIG] mask=${mask_root}"
        [[ -n "$gt_root" ]] && echo "[CONFIG] gt=${gt_root}"
    } >> "${log_path}"

    local gt_flag=""
    [[ -n "$gt_root" ]] && gt_flag="--gt_root $gt_root"
    local i3d_flag=""
    [[ -n "$I3D" && -f "$I3D" ]] && i3d_flag="--i3d_model_path $I3D"
    local raft_flag=""
    [[ -n "$RAFT" && -f "$RAFT" ]] && raft_flag="--raft_model_path $RAFT"

    local exit_code=0
    CUDA_VISIBLE_DEVICES="$gpu" python "${INFERENCE_DIR}/compare_all.py" \
      --dataset davis \
      --video_root "$vid_root" \
      --mask_root "$mask_root" \
      $gt_flag \
      --caption_yaml "$caption_yaml" \
      --ckpt "$ckpt" \
      --text_guidance_scale "$gs" \
      $extra_args \
      --neighbor_length 25 --subvideo_length 80 \
      --mask_dilation_iter "$dil" \
      $blend_flag \
      --base_model_path "$BASE_MODEL" \
      --vae_path "$VAE" \
      --diffueraser_path "$diffueraser_path" \
      --propainter_model_dir "$PROPAINTER" \
      --pcm_weights_path "$PCM" \
      $i3d_flag \
      $raft_flag \
      $eval_flag \
      --output_dir "${EXP_ROOT}/${outdir}" \
      --max_videos "$MAX_VIDEOS" \
      >> "${log_path}" 2>&1 || exit_code=$?

    if (( exit_code != 0 )); then
        echo "[FAIL] ${outdir} exited with code ${exit_code} on GPU ${gpu}" | tee -a "${log_path}"
        local target_dir="${EXP_ROOT}/${outdir}"
        if [[ -d "$target_dir" ]]; then
            local summaries
            summaries="$(find "$target_dir" -name "summary.json" 2>/dev/null | wc -l)"
            if (( summaries == 0 )); then
                rm -rf "$target_dir"
                echo "[CLEANUP] Removed incomplete dir: ${target_dir}" | tee -a "${log_path}"
            fi
        fi
    else
        echo "[DONE] ${outdir} on GPU ${gpu} ($(date))" | tee -a "${log_path}"
    fi

    release_gpu "$gpu"
    return "$exit_code"
}

declare -a EXP_CONFIGS=(
    "OR|2-Step|0.0|false|0|s2_OR_noblend_nodil_gs0.0|s2_OR_gs0"
    "BR|2-Step|0.0|false|0|s2_BR_noblend_nodil_gs0.0|s2_BR_gs0"
    "OR|4-Step|0.0|false|0|s4_OR_noblend_nodil_gs0.0|s4_OR_gs0"
    "BR|4-Step|0.0|false|0|s4_BR_noblend_nodil_gs0.0|s4_BR_gs0"
    "OR|2-Step|0.0|true|8|s2_OR_blend_dil8_gs0.0|s2_OR_b_gs0"
    "BR|2-Step|0.0|true|8|s2_BR_blend_dil8_gs0.0|s2_BR_b_gs0"
    "OR|4-Step|0.0|true|8|s4_OR_blend_dil8_gs0.0|s4_OR_b_gs0"
    "BR|4-Step|0.0|true|8|s4_BR_blend_dil8_gs0.0|s4_BR_b_gs0"
)

declare -a WEIGHT_SETS=(
    "Orign|${DIFFUERASER_ORIG}"
    "FT_S2_8K|${DIFFUERASER_FT_8K}"
    "FT_S2_26K|${DIFFUERASER_FT_26K}"
    "FT_S2_34K|${DIFFUERASER_FT_34K}"
    "FT_S2_48K|${DIFFUERASER_FT_48K}"
)

declare -a ACTIVE_WEIGHTS=()
for weight_line in "${WEIGHT_SETS[@]}"; do
    IFS='|' read -r weight_name weight_path <<< "$weight_line"
    if [[ -d "$weight_path" ]]; then
        ACTIVE_WEIGHTS+=("$weight_line")
    else
        echo "[SKIP WEIGHT] ${weight_name}: ${weight_path} not found"
    fi
done

require_dir "base model" "$BASE_MODEL" || exit 2
require_dir "vae" "$VAE" || exit 2
require_dir "propainter weights" "$PROPAINTER" || exit 2
require_dir "pcm weights" "$PCM" || exit 2
require_dir "OR video root" "$OR_VIDEO" || exit 2
require_dir "OR mask root" "$OR_MASK" || exit 2
require_dir "BR video root" "$BR_VIDEO" || exit 2
require_dir "BR mask root" "$BR_MASK" || exit 2

if (( ${#ACTIVE_WEIGHTS[@]} == 0 )); then
    echo "[ERROR] No DiffuEraser weight directories found."
    exit 2
fi

TOTAL=$(( ${#EXP_CONFIGS[@]} * ${#ACTIVE_WEIGHTS[@]} ))

echo "============================================================"
echo "  DiffuEraser Weight Sweep Runner"
echo "  $(date)"
echo "  Project : ${PROJECT_ROOT}"
echo "  Output  : ${EXP_ROOT}"
echo "  Logs    : ${LOG_DIR}"
echo "  GPUs    : ${ALL_GPUS[*]}  (min free: ${MIN_FREE_MB}MB)"
echo "  Jobs    : ${#EXP_CONFIGS[@]} configs x ${#ACTIVE_WEIGHTS[@]} weights = ${TOTAL}"
echo "============================================================"

cleanup

run_all_experiments() {
    local retry_round=$1
    local queued=0
    local fail_count=0
    declare -A local_pids=()

    for weight_line in "${ACTIVE_WEIGHTS[@]}"; do
        IFS='|' read -r weight_name weight_path <<< "$weight_line"

        for exp_line in "${EXP_CONFIGS[@]}"; do
            IFS='|' read -r exp_type ckpt gs blend dil outdir_base logname_base <<< "$exp_line"

            local outdir="${weight_name}_${outdir_base}"
            local logname="${weight_name}_${logname_base}"

            if [[ -f "${EXP_ROOT}/${outdir}/summary.json" ]]; then
                echo "[SKIP] ${outdir} already complete"
                queued=$((queued + 1))
                continue
            fi
            if [[ -d "${EXP_ROOT}/${outdir}" ]]; then
                rm -rf "${EXP_ROOT:?}/${outdir}"
                echo "[CLEANUP] Removed incomplete dir: ${EXP_ROOT}/${outdir}"
            fi

            local gpu
            gpu="$(acquire_gpu)"
            queued=$((queued + 1))
            echo "[ASSIGN] ${queued}/${TOTAL}: ${outdir} -> GPU ${gpu} (round ${retry_round})"

            run_experiment "$gpu" "$exp_type" "$ckpt" "$gs" "$blend" "$dil" "$outdir" "$logname" "$weight_path" &
            local_pids["$outdir"]=$!

            sleep 5
        done
    done

    echo ""
    echo "[INFO] All experiments assigned for round ${retry_round}. Waiting..."
    for outdir in "${!local_pids[@]}"; do
        local pid="${local_pids[$outdir]}"
        if ! wait "$pid"; then
            echo "[WARN] ${outdir} (PID ${pid}) failed in round ${retry_round}"
            fail_count=$((fail_count + 1))
        fi
    done

    cleanup
    return "$fail_count"
}

for attempt in $(seq 1 "$MAX_RETRIES"); do
    echo ""
    echo "============================================================"
    echo "  Round ${attempt} / ${MAX_RETRIES}  |  $(date)"
    echo "============================================================"

    run_all_experiments "$attempt"
    fail_count=$?

    if (( fail_count == 0 )); then
        echo "[OK] All experiments completed successfully in round ${attempt}."
        break
    fi

    echo "[WARN] ${fail_count} experiment(s) failed in round ${attempt}."
    if (( attempt < MAX_RETRIES )); then
        echo "[INFO] Retrying incomplete experiments in next round..."
        sleep 10
    fi
done

echo ""
echo "Generating final report..."
mapfile -t REPORT_DIRS < <(find "${EXP_ROOT}" -maxdepth 1 -type d \( -name "Orign_*" -o -name "FT_S2_*" -o -name "Finetune_*" \) | sort)
if (( ${#REPORT_DIRS[@]} > 0 )); then
    (
        cd "${PROJECT_ROOT}" || exit 1
        python "${INFERENCE_DIR}/generate_report.py" "${REPORT_DIRS[@]}"
    )
    mv "${PROJECT_ROOT}/experiment_report.md" "${EXP_ROOT}/experiment_report.md" 2>/dev/null || true
else
    echo "[WARN] No completed experiment directories found for report generation."
fi

echo ""
echo "============================================================"
echo "  DONE"
echo "  Results: ${EXP_ROOT}"
echo "  Report : ${EXP_ROOT}/experiment_report.md"
echo "  Logs   : ${LOG_DIR}"
echo "============================================================"
