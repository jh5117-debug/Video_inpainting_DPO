#!/bin/bash
#SBATCH --job-name=VDPO_VB
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=logs/videodpo-vc2-vbench-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="${PROJECT_NAME:-Video_inpainting_DPO}"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${PROJECT_DEV:-}" ]]; then
    PROJECT_ROOT="${PROJECT_DEV}/${PROJECT_NAME}"
  else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
export WORLDMODELPHY_PROCESS_NAME="${WORLDMODELPHY_PROCESS_NAME:-worldmodelphy}"
export PROCESS_TITLE="${PROCESS_TITLE:-${WORLDMODELPHY_PROCESS_NAME}}"

export PROJECT_HOME="${PROJECT_HOME:-/sc-projects/sc-proj-cc09-repair/hongyou}"
export PROJECT_DEV="${PROJECT_DEV:-${PROJECT_HOME}/dev}"
export PROJECT_DATA="${PROJECT_DATA:-${PROJECT_DEV}/data}"

VIDEODPO_REPO="${VIDEODPO_REPO:-${PROJECT_ROOT}/external/VideoDPO}"
VBENCH_ROOT="${VBENCH_ROOT:-${PROJECT_ROOT}/external/VBench}"
PROMPTS_FILE="${PROMPTS_FILE:-${VIDEODPO_REPO}/prompts/vbench_standard_prompts.txt}"
VBENCH_FULL_JSON="${VBENCH_FULL_JSON:-${VBENCH_ROOT}/vbench/VBench_full_info.json}"

# Comma-separated label:path pairs. Example:
#   CKPT_SPECS="vc2_base:/path/model.ckpt,vc2_dpo:/path/last.ckpt"
CKPT_SPECS="${CKPT_SPECS:-vc2_base:${VIDEODPO_REPO}/checkpoints/vc2/model.ckpt}"

OUT_ROOT="${OUT_ROOT:-${PROJECT_DATA}/experiments/videodpo_vbench/$(date -u +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-${VIDEODPO_REPO}/configs/inference/inference_t2v_512_v2.0.yaml}"
HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-512}"
FRAMES="${FRAMES:--1}"
FPS="${FPS:-28}"
SAVE_FPS="${SAVE_FPS:-10}"
DDIM_STEPS="${DDIM_STEPS:-50}"
DDIM_ETA="${DDIM_ETA:-1.0}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-12.0}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-5}"
PROMPT_LIMIT="${PROMPT_LIMIT:-0}"
LINK_MODE="${LINK_MODE:-symlink}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_VBENCH="${RUN_VBENCH:-1}"
SEED_BASE="${SEED_BASE:-20230211}"
DIMENSIONS="${DIMENSIONS:-subject_consistency background_consistency aesthetic_quality imaging_quality object_class multiple_objects color spatial_relationship scene temporal_style overall_consistency human_action temporal_flickering motion_smoothness dynamic_degree appearance_style}"
PATCH_VIDEODPO_INFERENCE="${PATCH_VIDEODPO_INFERENCE:-1}"

CONDA_ENV="${CONDA_ENV:-${VIDEODPO_CONDA_ENV:-videodpo}}"
if [[ -n "${CONDA_BASE:-}" && -x "${CONDA_BASE}/bin/conda" ]]; then
  :
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$("${CONDA_EXE}" info --base)"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  echo "[videodpo-vbench][error] conda not found; set CONDA_EXE or CONDA_BASE." >&2
  exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if [[ -x "${CONDA_ENV}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_ENV}/bin/python"
  conda activate "${CONDA_ENV}" >/dev/null 2>&1 || true
else
  conda activate "${CONDA_ENV}"
  PYTHON_BIN="$(command -v python)"
fi

echo "[videodpo-vbench] python=${PYTHON_BIN}"
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys
print(f"[videodpo-vbench] sys.executable={sys.executable}")
missing = [m for m in ["torch", "pytorch_lightning", "omegaconf", "einops"] if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"[videodpo-vbench][error] missing required inference modules: {missing}")
PY

if [[ ! -d "${VIDEODPO_REPO}" ]]; then
  echo "[videodpo-vbench][error] VIDEODPO_REPO not found: ${VIDEODPO_REPO}" >&2
  echo "[videodpo-vbench][error] Run from ${PROJECT_ROOT}: git submodule update --init --recursive" >&2
  exit 1
fi
if [[ ! -f "${PROMPTS_FILE}" ]]; then
  echo "[videodpo-vbench][error] PROMPTS_FILE not found: ${PROMPTS_FILE}" >&2
  exit 1
fi
if [[ "${PATCH_VIDEODPO_INFERENCE}" == "1" ]]; then
  "${PYTHON_BIN}" - "${VIDEODPO_REPO}/scripts/inference.py" "${VIDEODPO_REPO}/scripts/inference_utils.py" <<'PY'
from pathlib import Path
import sys

inference_path = Path(sys.argv[1])
utils_path = Path(sys.argv[2])

text = inference_path.read_text()
old = 'parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")'
new = 'parser.add_argument("--savefps", type=int, default=10, help="video fps to generate")'
if old in text:
    inference_path.write_text(text.replace(old, new))
    print(f"[videodpo-vbench] patched savefps type in {inference_path}")
elif new in text:
    print(f"[videodpo-vbench] savefps type already patched in {inference_path}")
else:
    raise SystemExit(f"[videodpo-vbench][error] could not locate savefps parser line in {inference_path}")

utils_text = utils_path.read_text()
if "imageio.mimsave(savepath" in utils_text:
    print(f"[videodpo-vbench] video writer already patched in {utils_path}")
else:
    old_func = '''def save_videos(batch_tensors, savedir, filenames, fps=10):
    # b,samples,c,t,h,w
    n_samples = batch_tensors.shape[1]
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1.0, 1.0)
        video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w
        frame_grids = [
            torchvision.utils.make_grid(framesheet, nrow=int(n_samples))
            for framesheet in video
        ]  # [3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
        torchvision.io.write_video(
            savepath, grid, fps=fps, video_codec="h264", options={"crf": "10"}
        )
'''
    new_func = '''def save_videos(batch_tensors, savedir, filenames, fps=10):
    # b,samples,c,t,h,w
    fps = int(fps)
    n_samples = batch_tensors.shape[1]
    os.makedirs(savedir, exist_ok=True)
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1.0, 1.0)
        video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w
        frame_grids = [
            torchvision.utils.make_grid(framesheet, nrow=int(n_samples))
            for framesheet in video
        ]  # [3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        frames = grid.numpy()
        savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
        try:
            import imageio.v2 as imageio
            imageio.mimsave(savepath, list(frames), fps=fps, macro_block_size=1)
        except Exception as imageio_error:
            writer = cv2.VideoWriter(
                savepath,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (frames.shape[2], frames.shape[1]),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open cv2 VideoWriter for {savepath}") from imageio_error
            try:
                for frame in frames:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
'''
    if old_func not in utils_text:
        raise SystemExit(f"[videodpo-vbench][error] could not locate torchvision video writer in {utils_path}")
    utils_path.write_text(utils_text.replace(old_func, new_func))
    print(f"[videodpo-vbench] patched video writer in {utils_path}")
PY
fi
if [[ "${RUN_VBENCH}" == "1" && ! -f "${VBENCH_ROOT}/evaluate.py" ]]; then
  echo "[videodpo-vbench][error] VBench evaluate.py not found under VBENCH_ROOT=${VBENCH_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}" "${PROJECT_ROOT}/logs"
export PYTHONPATH="${VIDEODPO_REPO}:${VBENCH_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
RUN_PROMPTS_FILE="${PROMPTS_FILE}"
if [[ "${PROMPT_LIMIT}" != "0" ]]; then
  RUN_PROMPTS_FILE="${OUT_ROOT}/prompts_limit_${PROMPT_LIMIT}.txt"
  head -n "${PROMPT_LIMIT}" "${PROMPTS_FILE}" > "${RUN_PROMPTS_FILE}"
  if [[ "${RUN_VBENCH}" == "1" ]]; then
    echo "[videodpo-vbench][error] PROMPT_LIMIT is for inference smoke only." >&2
    echo "VBench standard mode expects the full VBench prompt suite. Set RUN_VBENCH=0 for smoke, or remove PROMPT_LIMIT for full evaluation." >&2
    exit 1
  fi
fi
echo "[videodpo-vbench] project_root=${PROJECT_ROOT}"
echo "[videodpo-vbench] videodpo_repo=${VIDEODPO_REPO}"
echo "[videodpo-vbench] vbench_root=${VBENCH_ROOT}"
echo "[videodpo-vbench] prompts=${RUN_PROMPTS_FILE}"
echo "[videodpo-vbench] out_root=${OUT_ROOT}"
echo "[videodpo-vbench] ckpt_specs=${CKPT_SPECS}"
echo "[videodpo-vbench] samples_per_prompt=${SAMPLES_PER_PROMPT} prompt_limit=${PROMPT_LIMIT}"

IFS=',' read -r -a CKPT_ITEMS <<< "${CKPT_SPECS}"
for item in "${CKPT_ITEMS[@]}"; do
  label="${item%%:*}"
  ckpt="${item#*:}"
  if [[ -z "${label}" || -z "${ckpt}" || "${label}" == "${ckpt}" ]]; then
    echo "[videodpo-vbench][error] Bad CKPT_SPECS item: ${item}" >&2
    exit 1
  fi
  if [[ ! -f "${ckpt}" ]]; then
    echo "[videodpo-vbench][error] checkpoint not found for ${label}: ${ckpt}" >&2
    exit 1
  fi

  raw_root="${OUT_ROOT}/${label}/raw"
  named_root="${OUT_ROOT}/${label}/vbench_standard_named"
  eval_root="${OUT_ROOT}/${label}/vbench_eval"
  mkdir -p "${raw_root}" "${named_root}" "${eval_root}"

  if [[ "${RUN_INFERENCE}" == "1" ]]; then
    for sample_idx in $(seq 0 $((SAMPLES_PER_PROMPT - 1))); do
      seed=$((SEED_BASE + sample_idx))
      sample_raw="${raw_root}/sample_${sample_idx}"
      mkdir -p "${sample_raw}"
      echo "[videodpo-vbench] infer label=${label} sample=${sample_idx} seed=${seed}"
      cd "${VIDEODPO_REPO}"
      "${PYTHON_BIN}" scripts/inference.py \
        --seed "${seed}" \
        --mode base \
        --ckpt_path "${ckpt}" \
        --config "${CONFIG}" \
        --savedir "${sample_raw}" \
        --n_samples 1 \
        --bs 1 \
        --height "${HEIGHT}" \
        --width "${WIDTH}" \
        --frames "${FRAMES}" \
        --unconditional_guidance_scale "${GUIDANCE_SCALE}" \
        --ddim_steps "${DDIM_STEPS}" \
        --ddim_eta "${DDIM_ETA}" \
        --prompt_file "${RUN_PROMPTS_FILE}" \
        --fps "${FPS}" \
        --savefps "${SAVE_FPS}"

      "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/videodpo_prepare_vbench_standard.py" \
        --raw_dir "${sample_raw}" \
        --prompts_file "${RUN_PROMPTS_FILE}" \
        --output_dir "${named_root}" \
        --sample_index "${sample_idx}" \
        --prompt_limit "${PROMPT_LIMIT}" \
        --mode "${LINK_MODE}" \
        --strict
    done
  fi

  if [[ "${RUN_VBENCH}" == "1" ]]; then
    echo "[videodpo-vbench] vbench label=${label}"
    cd "${VBENCH_ROOT}"
    "${PYTHON_BIN}" evaluate.py \
      --videos_path "${named_root}" \
      --output_path "${eval_root}" \
      --full_json_dir "${VBENCH_FULL_JSON}" \
      --dimension ${DIMENSIONS} \
      --mode vbench_standard

    latest_json="$(ls -t "${eval_root}"/results_*_eval_results.json | head -n 1)"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/summarize_vbench_results.py" "${latest_json}" \
      --output_json "${eval_root}/summary.json" \
      --output_csv "${eval_root}/summary.csv"
  fi
done

echo "[videodpo-vbench] done: ${OUT_ROOT}"
