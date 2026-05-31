#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

EXP_NAME="${EXP_NAME:-${RUN_NAME:-dpo_beta10_s1s2_4000}}"
PREFERENCE_MANIFEST="${PREFERENCE_MANIFEST:-}"
TRAIN_MASK_MODE="${TRAIN_MASK_MODE:-full}"
MASK_FROM_MANIFEST="${MASK_FROM_MANIFEST:-false}"
LOSS_REGION_MODE="${LOSS_REGION_MODE:-full}"
BETA_DPO="${BETA_DPO:-10}"
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-4000}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-4000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${PROJECT_ROOT}/weights}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${OUTPUT_ROOT}/experiments}"
NUM_GPUS="${NUM_GPUS:-8}"
BASE_QUAL30_DIR="${BASE_QUAL30_DIR:-}"
BASELINE_WEIGHTS_PATH="${BASELINE_WEIGHTS_PATH:-${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000}"
PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/external/VideoDPO/prompts/vbench_standard_prompts.txt}"
QUAL30_SEED="${QUAL30_SEED:-42}"
SKIP_FULL_VBENCH="${SKIP_FULL_VBENCH:-false}"
SKIP_QUAL30="${SKIP_QUAL30:-false}"

VAL_STEPS="${VAL_STEPS:-999999}"
CKPT_STEPS="${CKPT_STEPS:-1000}"
CKPT_LIMIT="${CKPT_LIMIT:-2}"
REPORT_TO="${REPORT_TO:-none}"
DPO_DIAG_SAVE_WANDB="${DPO_DIAG_SAVE_WANDB:-false}"
ENABLE_DPO_DIAG="${ENABLE_DPO_DIAG:-true}"
DPO_DIAG_LOG_EVERY="${DPO_DIAG_LOG_EVERY:-10}"
DPO_DIAG_SAVE_CSV="${DPO_DIAG_SAVE_CSV:-true}"
LINGBOT_PROCESS_NAME="${LINGBOT_PROCESS_NAME:-lingbot-worldphy}"
WORLDMODELPHY_PROCESS_NAME="${WORLDMODELPHY_PROCESS_NAME:-${LINGBOT_PROCESS_NAME}}"
PROCESS_TITLE="${PROCESS_TITLE:-${LINGBOT_PROCESS_NAME}}"

TRAIN_HEIGHT="${TRAIN_HEIGHT:-320}"
TRAIN_WIDTH="${TRAIN_WIDTH:-512}"
RESOLUTION="${RESOLUTION:-512}"
NFRAMES="${NFRAMES:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
VAE_DTYPE="${VAE_DTYPE:-auto}"
POLICY_DTYPE="${POLICY_DTYPE:-auto}"
REF_DTYPE="${REF_DTYPE:-auto}"
TEXT_DTYPE="${TEXT_DTYPE:-auto}"
SFT_REG_WEIGHT="${SFT_REG_WEIGHT:-0.0}"
LOSE_GAP_WEIGHT="${LOSE_GAP_WEIGHT:-${DPO_LOSE_GAP_WEIGHT:-1.0}}"
DPO_LOSE_GAP_WEIGHT="${LOSE_GAP_WEIGHT}"
WINNER_ABS_REG_WEIGHT="${WINNER_ABS_REG_WEIGHT:-0.0}"
WINNER_GAP_REG_WEIGHT="${WINNER_GAP_REG_WEIGHT:-0.0}"
WINNER_GAP_REG_MARGIN="${WINNER_GAP_REG_MARGIN:-0.0}"
WINNER_GAP_REG_MODE="${WINNER_GAP_REG_MODE:-relu}"
LR="${LR:-1e-6}"
LR_SCHEDULER="${LR_SCHEDULER:-constant}"
LR_WARMUP="${LR_WARMUP:-500}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-42}"
DAVIS_OVERSAMPLE="${DAVIS_OVERSAMPLE:-10}"
VIDEODPO_FRAME_STRIDE="${VIDEODPO_FRAME_STRIDE:-1}"
VIDEODPO_CLIP_LENGTH="${VIDEODPO_CLIP_LENGTH:-1.0}"
VIDEODPO_FULL_MASK_VALUE="${VIDEODPO_FULL_MASK_VALUE:-0.0}"
VAL_NUM_INFERENCE_STEPS="${VAL_NUM_INFERENCE_STEPS:-6}"
VAL_MASK_DILATION_ITER="${VAL_MASK_DILATION_ITER:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
SPLIT_POS_NEG_FORWARD="${SPLIT_POS_NEG_FORWARD:-1}"
CHUNK_ALIGNED="${CHUNK_ALIGNED:-1}"
USE_8BIT_ADAM="${USE_8BIT_ADAM:-0}"
XFORMERS="${XFORMERS:-0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_PREFIX:-diffueraser}}"
VBENCH_CONDA_ENV="${VBENCH_CONDA_ENV:-${CONDA_ENV}}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${WEIGHTS_DIR}/stable-diffusion-v1-5}"
VAE_PATH="${VAE_PATH:-${WEIGHTS_DIR}/sd-vae-ft-mse}"
REF_MODEL_PATH="${REF_MODEL_PATH:-${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000}"
BASELINE_UNET_PATH="${BASELINE_UNET_PATH:-${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000}"

STAGE1_SCRIPT="${STAGE1_SCRIPT:-${PROJECT_ROOT}/training/dpo/scripts/03_dpo_stage1.sbatch}"
STAGE2_SCRIPT="${STAGE2_SCRIPT:-${PROJECT_ROOT}/training/dpo/scripts/04_dpo_stage2.sbatch}"
VBENCH_RUNNER="${VBENCH_RUNNER:-${PROJECT_ROOT}/DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh}"
VBENCH_ROOT="${VBENCH_ROOT:-${PROJECT_ROOT}/external/VBench}"

ts() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[PIPELINE $(ts)] $*"
}

die() {
  echo "[PIPELINE][ERROR] $*" >&2
  exit 1
}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

sanitize_beta() {
  echo "$1" | sed 's/\./p/g; s/[^A-Za-z0-9_.-]/_/g'
}

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "$path" ]] || die "${label} not found: ${path}"
}

resolve_python() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    echo "${PYTHON_BIN}"
  elif [[ -d "${CONDA_ENV}" && -x "${CONDA_ENV}/bin/python" ]]; then
    echo "${CONDA_ENV}/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    command -v python3
  elif command -v python >/dev/null 2>&1; then
    command -v python
  else
    die "python not found; set PYTHON_BIN or CONDA_ENV/CONDA_ENV_PREFIX"
  fi
}

PIPELINE_TS="${PIPELINE_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_VERSION="${RUN_VERSION:-${PIPELINE_TS}}"
BETA_LABEL="$(sanitize_beta "${BETA_DPO}")"
PIPELINE_ROOT="${OUTPUT_ROOT}/logs/pipelines/${EXP_NAME}/${PIPELINE_TS}"
TRAIN_LOG_DIR="${OUTPUT_ROOT}/logs/train/${EXP_NAME}"
REPORT_DIR="${OUTPUT_ROOT}/reports"
mkdir -p "${PIPELINE_ROOT}" "${TRAIN_LOG_DIR}" "${REPORT_DIR}"

STAGE1_RUN_NAME="${EXP_NAME}_stage1"
STAGE2_RUN_NAME="${EXP_NAME}_stage2"
STAGE1_RUN_DIR="${EXPERIMENTS_DIR}/dpo/stage1/${RUN_VERSION}_${STAGE1_RUN_NAME}"
STAGE2_RUN_DIR="${EXPERIMENTS_DIR}/dpo/stage2/${RUN_VERSION}_${STAGE2_RUN_NAME}"
STAGE1_LOG="${TRAIN_LOG_DIR}/stage1.log"
STAGE2_LOG="${TRAIN_LOG_DIR}/stage2.log"

log "Precheck"
require_path "${PREFERENCE_MANIFEST}" "PREFERENCE_MANIFEST"
require_path "${BASE_MODEL_PATH}" "base model"
require_path "${VAE_PATH}" "VAE"
require_path "${REF_MODEL_PATH}" "DiffuEraser converted weights"
require_path "${STAGE1_SCRIPT}" "Stage1 script"
require_path "${STAGE2_SCRIPT}" "Stage2 script"
require_path "${VBENCH_RUNNER}" "VBench runner"
require_path "${PROMPTS_FILE}" "VBench prompts"
require_path "${VBENCH_ROOT}/evaluate.py" "VBench evaluate.py"
require_path "${BASELINE_WEIGHTS_PATH}" "DiffuEraser-base weights"

PYTHON_BIN="$(resolve_python)"
log "Using python: ${PYTHON_BIN}"

FFMPEG_BIN="${FFMPEG_BIN:-$(command -v ffmpeg || true)}"
if [[ -z "${FFMPEG_BIN}" ]]; then
  FFMPEG_BIN="$("${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
try:
    import imageio_ffmpeg
    print(imageio_ffmpeg.get_ffmpeg_exe())
except Exception:
    pass
PY
)"
fi
if [[ -z "${FFMPEG_BIN}" ]]; then
  log "ffmpeg not found in PATH and imageio_ffmpeg could not be resolved yet; qual30 side-by-side will fail if no ffmpeg is available after generation."
else
  log "Using ffmpeg: ${FFMPEG_BIN}"
fi

cat <<EOF
============================================================
  DPO beta10 two-stage pipeline
============================================================
  Exp Name:              ${EXP_NAME}
  Project Root:          ${PROJECT_ROOT}
  Output Root:           ${OUTPUT_ROOT}
  Experiments Dir:       ${EXPERIMENTS_DIR}
  Preference Manifest:   ${PREFERENCE_MANIFEST}
  Train Mask Mode:       ${TRAIN_MASK_MODE}
  Mask From Manifest:    ${MASK_FROM_MANIFEST}
  Loss Region Mode:      ${LOSS_REGION_MODE}
  Beta DPO:              ${BETA_DPO}
  Lose Gap Weight:       ${DPO_LOSE_GAP_WEIGHT}
  SFT Reg Weight:        ${SFT_REG_WEIGHT}
  Winner Abs Reg Weight: ${WINNER_ABS_REG_WEIGHT}
  Winner Gap Reg Weight: ${WINNER_GAP_REG_WEIGHT}
  Winner Gap Reg Margin: ${WINNER_GAP_REG_MARGIN}
  Winner Gap Reg Mode:   ${WINNER_GAP_REG_MODE}
  Stage1 Steps:          ${STAGE1_MAX_STEPS}
  Stage2 Steps:          ${STAGE2_MAX_STEPS}
  Num GPUs:              ${NUM_GPUS}
  Qual30 Seed:           ${QUAL30_SEED}
  Skip Qual30:           ${SKIP_QUAL30}
  Skip Full VBench:      ${SKIP_FULL_VBENCH}
============================================================
EOF

export PIPELINE_ROOT EXP_NAME PROJECT_ROOT OUTPUT_ROOT EXPERIMENTS_DIR RUN_VERSION
export PREFERENCE_MANIFEST TRAIN_MASK_MODE MASK_FROM_MANIFEST LOSS_REGION_MODE BETA_DPO
export STAGE1_MAX_STEPS STAGE2_MAX_STEPS STAGE1_RUN_NAME STAGE2_RUN_NAME
export STAGE1_RUN_DIR STAGE2_RUN_DIR STAGE1_LOG STAGE2_LOG
export SKIP_QUAL30 SKIP_FULL_VBENCH PROMPTS_FILE QUAL30_SEED
export WEIGHTS_DIR BASE_MODEL_PATH VAE_PATH REF_MODEL_PATH BASELINE_UNET_PATH
export NUM_GPUS VAL_STEPS CKPT_STEPS CKPT_LIMIT REPORT_TO DPO_DIAG_SAVE_WANDB
export FFMPEG_BIN PYTHON_BIN
export ENABLE_DPO_DIAG DPO_DIAG_LOG_EVERY DPO_DIAG_SAVE_CSV LINGBOT_PROCESS_NAME
export WORLDMODELPHY_PROCESS_NAME PROCESS_TITLE TRAIN_HEIGHT TRAIN_WIDTH RESOLUTION NFRAMES
export NUM_WORKERS LOGGING_STEPS MIXED_PRECISION VAE_DTYPE POLICY_DTYPE REF_DTYPE TEXT_DTYPE
export SFT_REG_WEIGHT LOSE_GAP_WEIGHT DPO_LOSE_GAP_WEIGHT WINNER_ABS_REG_WEIGHT
export WINNER_GAP_REG_WEIGHT WINNER_GAP_REG_MARGIN WINNER_GAP_REG_MODE
export LR LR_SCHEDULER LR_WARMUP GRAD_ACCUM BATCH_SIZE
export SEED DAVIS_OVERSAMPLE VIDEODPO_FRAME_STRIDE VIDEODPO_CLIP_LENGTH VIDEODPO_FULL_MASK_VALUE
export VAL_NUM_INFERENCE_STEPS VAL_MASK_DILATION_ITER GRADIENT_CHECKPOINTING SPLIT_POS_NEG_FORWARD
export CHUNK_ALIGNED USE_8BIT_ADAM XFORMERS MAIN_PROCESS_PORT WANDB_ENTITY CONDA_ENV VBENCH_CONDA_ENV

"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

keys = [
    "EXP_NAME", "PROJECT_ROOT", "OUTPUT_ROOT", "EXPERIMENTS_DIR",
    "PREFERENCE_MANIFEST", "TRAIN_MASK_MODE", "MASK_FROM_MANIFEST",
    "LOSS_REGION_MODE", "BETA_DPO", "STAGE1_MAX_STEPS", "STAGE2_MAX_STEPS",
    "LOSE_GAP_WEIGHT", "DPO_LOSE_GAP_WEIGHT", "SFT_REG_WEIGHT",
    "WINNER_ABS_REG_WEIGHT", "WINNER_GAP_REG_WEIGHT",
    "WINNER_GAP_REG_MARGIN", "WINNER_GAP_REG_MODE",
    "STAGE1_RUN_DIR", "STAGE2_RUN_DIR", "WEIGHTS_DIR", "NUM_GPUS",
    "SKIP_QUAL30", "SKIP_FULL_VBENCH",
]
manifest = {k: os.environ.get(k, "") for k in keys}
Path(os.environ["PIPELINE_ROOT"]).mkdir(parents=True, exist_ok=True)
(Path(os.environ["PIPELINE_ROOT"]) / "pipeline_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)
PY

log "Stage1 start: ${STAGE1_RUN_NAME}"
(
  export RUN_NAME="${STAGE1_RUN_NAME}"
  export RUN_VERSION="${RUN_VERSION}"
  export MAX_STEPS="${STAGE1_MAX_STEPS}"
  export DPO_DATASET_TYPE="generated_loser_manifest"
  export CKPT_STEPS CKPT_LIMIT VAL_STEPS
  export DPO_DATA_ROOT="${DPO_DATA_ROOT:-${OUTPUT_ROOT}/DPO_Finetune_data}"
  export BASELINE_UNET_PATH
  bash "${STAGE1_SCRIPT}"
) > "${STAGE1_LOG}" 2>&1 || {
  log "Stage1 failed; see ${STAGE1_LOG}"
  exit 1
}

log "Stage1 complete: ${STAGE1_RUN_DIR}"
require_path "${STAGE1_RUN_DIR}/last_weights/unet_main" "Stage1 unet_main last_weights"
require_path "${STAGE1_RUN_DIR}/last_weights/brushnet" "Stage1 brushnet last_weights"

log "Stage2 start: ${STAGE2_RUN_NAME}"
(
  export RUN_NAME="${STAGE2_RUN_NAME}"
  export RUN_VERSION="${RUN_VERSION}"
  export MAX_STEPS="${STAGE2_MAX_STEPS}"
  export DPO_DATASET_TYPE="generated_loser_manifest"
  export PRETRAINED_DPO_S1="${STAGE1_RUN_DIR}/last_weights"
  export STAGE1_WEIGHTS_DIR="${PRETRAINED_DPO_S1}"
  export CKPT_STEPS CKPT_LIMIT VAL_STEPS
  export DPO_DATA_ROOT="${DPO_DATA_ROOT:-${OUTPUT_ROOT}/DPO_Finetune_data}"
  bash "${STAGE2_SCRIPT}"
) > "${STAGE2_LOG}" 2>&1 || {
  log "Stage2 failed; see ${STAGE2_LOG}"
  exit 1
}

log "Stage2 complete: ${STAGE2_RUN_DIR}"
require_path "${STAGE2_RUN_DIR}/last_weights/unet_main" "Stage2 unet_main last_weights"
require_path "${STAGE2_RUN_DIR}/last_weights/brushnet" "Stage2 brushnet last_weights"

QUAL_ROOT="${OUTPUT_ROOT}/logs/qual_sbs_30/${EXP_NAME}_${PIPELINE_TS}"
QUAL_PROMPTS="${QUAL_ROOT}/prompts_qual30_seed${QUAL30_SEED}.txt"
QUAL_BASE_OUT="${QUAL_ROOT}/base"
QUAL_EXP_OUT="${QUAL_ROOT}/exp"
QUAL_SBS_OUT="${QUAL_ROOT}/side_by_side"
QUAL_BASE_VIDEO_DIR=""
QUAL_EXP_VIDEO_DIR="${QUAL_EXP_OUT}/vbench_standard_named"

if ! is_true "${SKIP_QUAL30}"; then
  log "Qual30 start"
  mkdir -p "${QUAL_ROOT}" "${QUAL_BASE_OUT}" "${QUAL_EXP_OUT}" "${QUAL_SBS_OUT}"
  export PROMPTS_FILE QUAL_PROMPTS QUAL30_SEED
  "${PYTHON_BIN}" - <<'PY'
import os
import random
from pathlib import Path

src = Path(os.environ["PROMPTS_FILE"])
dst = Path(os.environ["QUAL_PROMPTS"])
seed = int(os.environ["QUAL30_SEED"])
prompts = []
seen = set()
for line in src.read_text(encoding="utf-8").splitlines():
    prompt = line.strip()
    if prompt and prompt not in seen:
        prompts.append(prompt)
        seen.add(prompt)
random.Random(seed).shuffle(prompts)
chosen = prompts[:30]
if len(chosen) < 30:
    raise SystemExit(f"only {len(chosen)} unique prompts available")
dst.write_text("\n".join(chosen) + "\n", encoding="utf-8")
print(f"[qual30] prompts={len(chosen)} path={dst}")
PY

  if [[ -n "${BASE_QUAL30_DIR}" ]]; then
    case "${BASE_QUAL30_DIR,,}" in
      *vc2*)
        log "Ignoring BASE_QUAL30_DIR because it looks like VC2, not DiffuEraser-base: ${BASE_QUAL30_DIR}"
        ;;
      *)
        if [[ -d "${BASE_QUAL30_DIR}" ]]; then
          export BASE_QUAL30_DIR QUAL_PROMPTS
          if "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

base = Path(os.environ["BASE_QUAL30_DIR"])
prompts = [p.strip() for p in Path(os.environ["QUAL_PROMPTS"]).read_text(encoding="utf-8").splitlines() if p.strip()]
missing = [str(base / f"{p}-0.mp4") for p in prompts if not (base / f"{p}-0.mp4").exists()]
if missing:
    print(f"[qual30] existing base missing={len(missing)}")
    raise SystemExit(1)
print("[qual30] existing DiffuEraser-base qual30 matches prompts")
PY
          then
            QUAL_BASE_VIDEO_DIR="${BASE_QUAL30_DIR}"
          fi
        fi
        ;;
    esac
  fi

  if [[ -z "${QUAL_BASE_VIDEO_DIR}" ]]; then
    log "Generating DiffuEraser-base qual30"
    (
      export WEIGHTS_PATH="${BASELINE_WEIGHTS_PATH}"
      export OUT_ROOT="${QUAL_BASE_OUT}"
      export PROMPTS_FILE="${QUAL_PROMPTS}"
      export GENERATE=1
      export RUN_VBENCH=0
      export SAMPLES_PER_PROMPT=1
      export PROMPT_LIMIT=0
      export GEN_STAGE=stage2
      export HEIGHT="${TRAIN_HEIGHT}"
      export WIDTH="${TRAIN_WIDTH}"
      export FRAMES="${NFRAMES}"
      export CONDA_ENV VBENCH_CONDA_ENV PROJECT_ROOT WEIGHTS_DIR BASE_MODEL_PATH VAE_PATH
      bash "${VBENCH_RUNNER}"
    ) > "${QUAL_ROOT}/base_generation.log" 2>&1
    QUAL_BASE_VIDEO_DIR="${QUAL_BASE_OUT}/vbench_standard_named"
  fi

  log "Generating ${EXP_NAME} qual30"
  (
    export WEIGHTS_PATH="${STAGE2_RUN_DIR}/last_weights"
    export OUT_ROOT="${QUAL_EXP_OUT}"
    export PROMPTS_FILE="${QUAL_PROMPTS}"
    export GENERATE=1
    export RUN_VBENCH=0
    export SAMPLES_PER_PROMPT=1
    export PROMPT_LIMIT=0
    export GEN_STAGE=stage2
    export HEIGHT="${TRAIN_HEIGHT}"
    export WIDTH="${TRAIN_WIDTH}"
    export FRAMES="${NFRAMES}"
    export CONDA_ENV VBENCH_CONDA_ENV PROJECT_ROOT WEIGHTS_DIR BASE_MODEL_PATH VAE_PATH
    bash "${VBENCH_RUNNER}"
  ) > "${QUAL_ROOT}/exp_generation.log" 2>&1

  export QUAL_BASE_VIDEO_DIR QUAL_EXP_VIDEO_DIR QUAL_SBS_OUT QUAL_ROOT EXP_NAME
  "${PYTHON_BIN}" - <<'PY'
import csv
import html
import os
import shutil
import subprocess
from pathlib import Path

base = Path(os.environ["QUAL_BASE_VIDEO_DIR"])
exp = Path(os.environ["QUAL_EXP_VIDEO_DIR"])
out = Path(os.environ["QUAL_SBS_OUT"])
root = Path(os.environ["QUAL_ROOT"])
name = os.environ["EXP_NAME"]
ffmpeg = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
if not ffmpeg:
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise SystemExit(f"ffmpeg not available for side-by-side generation: {exc}") from exc
out.mkdir(parents=True, exist_ok=True)

base_files = {p.name: p for p in base.glob("*.mp4")}
exp_files = {p.name: p for p in exp.glob("*.mp4")}
common = sorted(set(base_files) & set(exp_files))
if len(common) < 30:
    raise SystemExit(f"expected 30 paired qual videos, found {len(common)}")

with (root / "pair_manifest.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "diffueraser_base_video", "experiment_video", "side_by_side_video"])
    for filename in common:
        dst = out / filename
        writer.writerow([filename, base_files[filename], exp_files[filename], dst])
        cmd = [
            ffmpeg, "-y",
            "-i", str(base_files[filename]),
            "-i", str(exp_files[filename]),
            "-filter_complex",
            "[0:v]scale=512:320,setsar=1,drawtext=text='DiffuEraser-base':x=12:y=12:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.55[v0];"
            f"[1:v]scale=512:320,setsar=1,drawtext=text='{name}':x=12:y=12:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.55[v1];"
            "[v0][v1]hstack=inputs=2[v]",
            "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            str(dst),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

with (root / "index.html").open("w", encoding="utf-8") as f:
    f.write("<html><body><h1>DiffuEraser-base vs experiment qual30</h1>\n")
    f.write(f"<p>experiment={html.escape(name)} paired={len(common)}</p>\n")
    f.write(f"<p>side_by_side_dir={html.escape(str(out))}</p>\n")
    for video in sorted(out.glob("*.mp4")):
        f.write(f"<div><h3>{html.escape(video.name)}</h3><video src='side_by_side/{html.escape(video.name)}' controls width='1024'></video></div>\n")
    f.write("</body></html>\n")

(root / "README.md").write_text(
    f"# {name} qual30 side-by-side\n\n"
    f"DiffuEraser-base videos: `{base}`\n"
    f"Experiment videos: `{exp}`\n"
    f"Side-by-side videos: `{out}`\n"
    f"Index: `{root / 'index.html'}`\n"
    f"Paired videos: {len(common)}\n",
    encoding="utf-8",
)
print(f"[qual30] side_by_side_dir={out}")
PY
else
  log "Qual30 skipped"
fi

FULL_VBENCH_ROOT="${OUTPUT_ROOT}/logs/vbench_${EXP_NAME}/${EXP_NAME}_full_vbench_${PIPELINE_TS}"
if ! is_true "${SKIP_FULL_VBENCH}"; then
  log "Full VBench start"
  (
    export WEIGHTS_PATH="${STAGE2_RUN_DIR}/last_weights"
    export OUT_ROOT="${FULL_VBENCH_ROOT}"
    export PROMPTS_FILE="${PROMPTS_FILE}"
    export GENERATE=1
    export RUN_VBENCH=1
    export SAMPLES_PER_PROMPT="${FULL_VBENCH_SAMPLES_PER_PROMPT:-5}"
    export PROMPT_LIMIT=0
    export GEN_STAGE=stage2
    export HEIGHT="${TRAIN_HEIGHT}"
    export WIDTH="${TRAIN_WIDTH}"
    export FRAMES="${NFRAMES}"
    export CONDA_ENV VBENCH_CONDA_ENV PROJECT_ROOT WEIGHTS_DIR BASE_MODEL_PATH VAE_PATH
    bash "${VBENCH_RUNNER}"
  ) > "${FULL_VBENCH_ROOT}.log" 2>&1

  require_path "${FULL_VBENCH_ROOT}/vbench_eval/summary.csv" "VBench summary.csv"
  require_path "${FULL_VBENCH_ROOT}/vbench_eval/summary.json" "VBench summary.json"
  export SUMMARY_CSV="${FULL_VBENCH_ROOT}/vbench_eval/summary.csv"
  export SCORE_TABLE="${FULL_VBENCH_ROOT}/vbench_score_table.md"
  "${PYTHON_BIN}" - <<'PY'
import csv
import os
from pathlib import Path

summary = Path(os.environ["SUMMARY_CSV"])
out = Path(os.environ["SCORE_TABLE"])
rows = list(csv.reader(summary.open(encoding="utf-8")))
with out.open("w", encoding="utf-8") as f:
    f.write("# VBench Score Table\n\n")
    f.write(f"source: `{summary}`\n\n")
    f.write("| " + " | ".join(rows[0]) + " |\n")
    f.write("| " + " | ".join(["---"] * len(rows[0])) + " |\n")
    for row in rows[1:]:
        f.write("| " + " | ".join(row) + " |\n")
PY
  cat > "${FULL_VBENCH_ROOT}/README.md" <<EOF
# ${EXP_NAME} full VBench

Stage2 weights: \`${STAGE2_RUN_DIR}/last_weights\`
Videos: \`${FULL_VBENCH_ROOT}/vbench_standard_named\`
Summary CSV: \`${FULL_VBENCH_ROOT}/vbench_eval/summary.csv\`
Summary JSON: \`${FULL_VBENCH_ROOT}/vbench_eval/summary.json\`
Score table: \`${FULL_VBENCH_ROOT}/vbench_score_table.md\`
EOF
else
  log "Full VBench skipped"
fi

REPORT_PATH="${REPORT_DIR}/${EXP_NAME}_pipeline_report.md"
{
  echo "# ${EXP_NAME} pipeline report"
  echo
  echo "status: success"
  echo "pipeline_root: \`${PIPELINE_ROOT}\`"
  echo "stage1_run_dir: \`${STAGE1_RUN_DIR}\`"
  echo "stage2_run_dir: \`${STAGE2_RUN_DIR}\`"
  echo "beta_dpo: ${BETA_DPO}"
  echo "lose_gap_weight: ${DPO_LOSE_GAP_WEIGHT}"
  echo "sft_reg_weight: ${SFT_REG_WEIGHT}"
  echo "winner_abs_reg_weight: ${WINNER_ABS_REG_WEIGHT}"
  echo "winner_gap_reg_weight: ${WINNER_GAP_REG_WEIGHT}"
  echo "winner_gap_reg_margin: ${WINNER_GAP_REG_MARGIN}"
  echo "winner_gap_reg_mode: ${WINNER_GAP_REG_MODE}"
  echo "stage1_max_steps: ${STAGE1_MAX_STEPS}"
  echo "stage2_max_steps: ${STAGE2_MAX_STEPS}"
  echo "stage1_dpo_diag: \`${STAGE1_RUN_DIR}/dpo_diagnostics.csv\`"
  echo "stage2_dpo_diag: \`${STAGE2_RUN_DIR}/dpo_diagnostics.csv\`"
  echo "stage1_log: \`${STAGE1_LOG}\`"
  echo "stage2_log: \`${STAGE2_LOG}\`"
  echo "qual30_side_by_side: \`${QUAL_SBS_OUT:-skipped}\`"
  echo "full_vbench_summary: \`${FULL_VBENCH_ROOT}/vbench_eval/summary.csv\`"
  echo "full_vbench_score_table: \`${FULL_VBENCH_ROOT}/vbench_score_table.md\`"
  echo
  if [[ -f "${FULL_VBENCH_ROOT}/vbench_score_table.md" ]]; then
    cat "${FULL_VBENCH_ROOT}/vbench_score_table.md"
  fi
} > "${REPORT_PATH}"

log "Pipeline complete"
log "Stage1 run dir: ${STAGE1_RUN_DIR}"
log "Stage2 run dir: ${STAGE2_RUN_DIR}"
log "Qual30 side-by-side dir: ${QUAL_SBS_OUT:-skipped}"
log "Full VBench root: ${FULL_VBENCH_ROOT}"
log "Report: ${REPORT_PATH}"
