#!/usr/bin/env bash
# PAI health check for the VideoDPO-data + DiffuEraser full-mask bridge.
#
# Default mode is non-destructive: inspect git/code/data/weights/env/GPU and run
# a lightweight dataset smoke when possible. Training smokes are opt-in:
#   RUN_FULLMASK_TRAIN_SMOKE=1 bash DPO_finetune/scripts/pai_videodpo_fullmask_health_check.sh
#   RUN_VIDEODPO_TRAIN_SMOKE=1 bash DPO_finetune/scripts/pai_videodpo_fullmask_health_check.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

PROJECT_HOME="${PROJECT_HOME:-/mnt/workspace/hj/nas_hj}"
PROJECT_DEV="${PROJECT_DEV:-${PROJECT_HOME}}"
PROJECT_DATA="${PROJECT_DATA:-${PROJECT_ROOT}}"
DATA="${DATA:-${PROJECT_ROOT}}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${PROJECT_ROOT}/weights}"
VIDEODPO_REPO="${VIDEODPO_REPO:-${PROJECT_ROOT}/external/VideoDPO}"
VBENCH_ROOT="${VBENCH_ROOT:-${PROJECT_ROOT}/external/VBench}"
DPO_DATA_ROOT="${DPO_DATA_ROOT:-${PROJECT_ROOT}/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
SMOKE_GPU="${SMOKE_GPU:-7}"

DIFFUERASER_ENV="${DIFFUERASER_ENV:-${CONDA_ENV_DIFFUERASER:-${CONDA_ENV:-}}}"
VIDEODPO_ENV="${VIDEODPO_ENV:-${CONDA_ENV_VIDEODPO:-}}"
RUN_DATASET_SMOKE="${RUN_DATASET_SMOKE:-1}"
RUN_FULLMASK_TRAIN_SMOKE="${RUN_FULLMASK_TRAIN_SMOKE:-0}"
RUN_VIDEODPO_TRAIN_SMOKE="${RUN_VIDEODPO_TRAIN_SMOKE:-0}"
STRICT="${STRICT:-0}"

FAILS=0
WARNS=0

section() {
  printf '\n========== %s ==========\n' "$1"
}

ok() {
  printf '[OK] %s\n' "$1"
}

warn() {
  WARNS=$((WARNS + 1))
  printf '[WARN] %s\n' "$1"
}

fail() {
  FAILS=$((FAILS + 1))
  printf '[FAIL] %s\n' "$1"
}

check_file() {
  local path="$1"
  local label="$2"
  if [[ -f "${path}" ]]; then
    ok "${label}: ${path}"
  else
    fail "${label} missing: ${path}"
  fi
}

check_file_any() {
  local label="$1"
  shift
  local path
  for path in "$@"; do
    if [[ -f "${path}" ]]; then
      ok "${label}: ${path}"
      return 0
    fi
  done
  fail "${label} missing; checked: $*"
}

check_dir() {
  local path="$1"
  local label="$2"
  if [[ -d "${path}" ]]; then
    ok "${label}: ${path}"
  else
    fail "${label} missing: ${path}"
  fi
}

find_env_python() {
  local requested="$1"
  shift
  local candidate

  if [[ -n "${requested}" ]]; then
    if [[ -x "${requested}/bin/python" ]]; then
      printf '%s\n' "${requested}/bin/python"
      return 0
    fi
    if [[ -x "${requested}" ]]; then
      printf '%s\n' "${requested}"
      return 0
    fi
  fi

  for candidate in "$@"; do
    if [[ -x "${candidate}/bin/python" ]]; then
      printf '%s\n' "${candidate}/bin/python"
      return 0
    fi
  done

  return 1
}

run_python_imports() {
  local label="$1"
  local py="$2"
  shift 2
  local modules=("$@")
  if [[ -z "${py}" || ! -x "${py}" ]]; then
    warn "${label}: python not found; skip import checks"
    return 0
  fi
  PYTHONPATH="${VIDEODPO_REPO}:${PROJECT_ROOT}:${PYTHONPATH:-}" "${py}" - "${modules[@]}" <<'PY'
import importlib
import sys

mods = sys.argv[1:]
failed = False
for mod in mods:
    try:
        m = importlib.import_module(mod)
        version = getattr(m, "__version__", "")
        suffix = f" {version}" if version else ""
        print(f"[OK] import {mod}{suffix}")
    except Exception as exc:
        failed = True
        print(f"[FAIL] import {mod}: {exc}")
sys.exit(1 if failed else 0)
PY
}

resolve_vc2_yaml() {
  local py="${1:-python3}"
PROJECT_ROOT="${PROJECT_ROOT}" \
PROJECT_HOME="${PROJECT_HOME}" \
PROJECT_DATA="${PROJECT_DATA}" \
DPO_DATA_ROOT="${DPO_DATA_ROOT}" \
VIDEODPO_REPO="${VIDEODPO_REPO}" \
DEEP_SCAN_DATA="${DEEP_SCAN_DATA:-0}" \
  "${py}" - <<'PY'
from pathlib import Path
import os
import re
import sys

def parse_meta(yaml_path: Path):
    text = yaml_path.read_text(encoding="utf-8")
    try:
        import yaml
        cfg = yaml.safe_load(text) or {}
        return [str(x) for x in cfg.get("META", [])]
    except Exception:
        metas = []
        in_meta = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("META:"):
                in_meta = True
                after = stripped.split(":", 1)[1].strip()
                metas.extend(re.findall(r"['\"]([^'\"]+)['\"]", after))
                continue
            if in_meta and stripped.startswith("-"):
                metas.append(stripped[1:].strip().strip("'\""))
            elif in_meta and stripped and not line.startswith((" ", "\t")):
                in_meta = False
        return metas

def root_is_valid(root: Path):
    return (root / "metadata.json").is_file() and (root / "pair.json").is_file()

raw = Path(os.environ["DPO_DATA_ROOT"]).expanduser()
repo = Path(os.environ["VIDEODPO_REPO"]).expanduser()
project_root = Path(os.environ["PROJECT_ROOT"]).expanduser()
project_home = Path(os.environ["PROJECT_HOME"]).expanduser()
project_data = Path(os.environ["PROJECT_DATA"]).expanduser()
deep_scan = os.environ.get("DEEP_SCAN_DATA", "0").lower() in {"1", "true", "yes", "on"}
candidates = []

if raw.exists():
    candidates.append(raw)

patterns = [
    project_root / "data/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml",
    project_root / "data/VideoDPO/configs/vc2_dpo/vidpro/train_data.yaml",
    repo / "configs/vc2_dpo/vidpro/train_data.yaml",
]
for item in patterns:
    candidates.append(item)

scan_yaml_roots = [project_root, project_data]
if deep_scan:
    scan_yaml_roots.append(project_home)

for root in scan_yaml_roots:
    if root.exists():
        candidates.extend(root.glob("**/train_data*.yaml"))

seen = set()
for candidate in candidates:
    candidate = candidate.expanduser()
    key = str(candidate)
    if key in seen:
        continue
    seen.add(key)
    if candidate.is_dir() and root_is_valid(candidate):
        print(candidate.resolve())
        sys.exit(0)
    if not candidate.is_file():
        continue
    roots = []
    ok = True
    for meta in parse_meta(candidate):
        meta_path = Path(meta).expanduser()
        if meta_path.is_absolute():
            root_candidates = [meta_path]
        else:
            root_candidates = [candidate.parent / meta_path, repo / meta_path, project_root / meta_path]
        match = next((p for p in root_candidates if root_is_valid(p)), None)
        if match is None:
            ok = False
            break
        roots.append(match.resolve())
    if ok and roots:
        print(candidate.resolve())
        sys.exit(0)

scan_pair_roots = [project_root, project_data]
if deep_scan:
    scan_pair_roots.append(project_home)

for root in scan_pair_roots:
    if not root.exists():
        continue
    for pair in root.glob("**/pair.json"):
        parent = pair.parent
        if root_is_valid(parent):
            print(parent.resolve())
            sys.exit(0)

sys.exit(2)
PY
}

section "PAI Health Check Context"
echo "project_root=${PROJECT_ROOT}"
echo "project_home=${PROJECT_HOME}"
echo "project_data=${PROJECT_DATA}"
echo "weights_dir=${WEIGHTS_DIR}"
echo "videodpo_repo=${VIDEODPO_REPO}"
echo "vbench_root=${VBENCH_ROOT}"
echo "dpo_data_root=${DPO_DATA_ROOT}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

section "Git"
if [[ -d "${PROJECT_ROOT}/.git" ]]; then
  git -C "${PROJECT_ROOT}" rev-parse --short HEAD >/dev/null 2>&1 && ok "git repo detected: $(git -C "${PROJECT_ROOT}" rev-parse --short HEAD)" || fail "git repo is present but not readable"
  git -C "${PROJECT_ROOT}" remote -v || warn "git remote not configured"
  if git -C "${PROJECT_ROOT}" status --short | grep -q .; then
    warn "working tree has local changes:"
    git -C "${PROJECT_ROOT}" status --short
  else
    ok "working tree clean"
  fi
  git -C "${PROJECT_ROOT}" diff --check >/tmp/pai_git_diff_check.$$ 2>&1
  if [[ $? -eq 0 ]]; then
    ok "git diff --check"
  else
    fail "git diff --check failed"
    cat /tmp/pai_git_diff_check.$$
  fi
  rm -f /tmp/pai_git_diff_check.$$
else
  warn "no .git directory under ${PROJECT_ROOT}; use the git-link commands from the handoff before pulling"
fi

section "Submodules And Code Layout"
check_dir "${VIDEODPO_REPO}" "VideoDPO repo"
check_file "${VIDEODPO_REPO}/scripts/train.py" "VideoDPO train.py"
check_file "${VIDEODPO_REPO}/configs/vc2_dpo/config.yaml" "VideoDPO VC2 config"
check_file "${VIDEODPO_REPO}/scripts/inference.py" "VideoDPO inference.py"
check_dir "${VBENCH_ROOT}" "VBench repo"
if [[ -d "${VBENCH_ROOT}" ]]; then
  check_file "${VBENCH_ROOT}/evaluate.py" "VBench evaluate.py"
fi

section "Static Code Checks"
if command -v rg >/dev/null 2>&1; then
  if rg -n '^(<<<<<<<|=======|>>>>>>>)' "${PROJECT_ROOT}/DPO_finetune/scripts" "${PROJECT_ROOT}/training/dpo" "${PROJECT_ROOT}/tools" "${PROJECT_ROOT}/patches" >/tmp/pai_conflicts.$$ 2>&1; then
    fail "conflict markers found"
    cat /tmp/pai_conflicts.$$
  else
    ok "no conflict markers in scripts/training/tools/patches"
  fi
  rm -f /tmp/pai_conflicts.$$
else
  warn "rg not installed; skip conflict-marker scan"
fi

if find "${PROJECT_ROOT}/DPO_finetune/scripts" -maxdepth 1 \( -name "*.sh" -o -name "*.sbatch" \) -print0 \
  | xargs -0 -r bash -n; then
  ok "bash syntax for DPO_finetune/scripts"
else
  fail "bash syntax check failed"
fi

if command -v python3 >/dev/null 2>&1; then
  if python3 -m py_compile \
    "${PROJECT_ROOT}/tools/smoke_videodpo_fullmask_dataset.py" \
    "${PROJECT_ROOT}/tools/prepare_videodpo_vc2_dataset.py" \
    "${PROJECT_ROOT}/tools/generate_diffueraser_fullmask_vbench.py" \
    "${PROJECT_ROOT}/tools/summarize_vbench_results.py" \
    "${PROJECT_ROOT}/training/dpo/dataset/videodpo_fullmask_dataset.py" \
    "${PROJECT_ROOT}/training/dpo/dataset/factory.py" \
    "${PROJECT_ROOT}/training/dpo/train_stage1.py" \
    "${PROJECT_ROOT}/training/dpo/train_stage2.py"; then
    ok "python syntax for key tools/training files"
  else
    fail "python syntax check failed"
  fi
else
  warn "python3 not found; skip py_compile"
fi

section "Full-Mask Setting Guard"
FULLMASK_WRAPPER="${PROJECT_ROOT}/DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage1.sbatch"
if [[ -f "${FULLMASK_WRAPPER}" ]]; then
  declare -a required_fullmask_settings=(
    'export TRAIN_HEIGHT="${TRAIN_HEIGHT:-320}"'
    'export TRAIN_WIDTH="${TRAIN_WIDTH:-512}"'
    'export MIXED_PRECISION="${MIXED_PRECISION:-no}"'
    'REQUESTED_SPLIT_POS_NEG_FORWARD="${SPLIT_POS_NEG_FORWARD:-0}"'
    'export BETA_DPO="${BETA_DPO:-5000}"'
    'export LR="${LR:-6e-6}"'
    'export CKPT_STEPS="${CKPT_STEPS:-499}"'
    'export NUM_WORKERS="${NUM_WORKERS:-16}"'
  )
  for setting in "${required_fullmask_settings[@]}"; do
    if grep -Fq "${setting}" "${FULLMASK_WRAPPER}"; then
      ok "full-mask wrapper setting: ${setting}"
    else
      fail "full-mask wrapper is not aligned: missing ${setting}"
    fi
  done
else
  fail "full-mask wrapper missing: ${FULLMASK_WRAPPER}"
fi

section "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || fail "nvidia-smi -L failed"
else
  fail "nvidia-smi not found"
fi

section "Environment Discovery"
DIFF_PY="$(find_env_python "${DIFFUERASER_ENV}" \
  "${PROJECT_ROOT}/conda_envs/diffueraser" \
  "${PROJECT_HOME}/conda_envs/diffueraser" \
  "/mnt/workspace/hj/conda_envs/diffueraser" \
  "/home/nvme01/conda_envs/diffueraser" || true)"
VIDEO_PY="$(find_env_python "${VIDEODPO_ENV}" \
  "${PROJECT_ROOT}/conda_envs/videodpo" \
  "${PROJECT_HOME}/conda_envs/videodpo" \
  "/mnt/workspace/hj/conda_envs/videodpo" \
  "/home/nvme01/conda_envs/videodpo" || true)"

if [[ -n "${DIFF_PY}" ]]; then
  ok "DiffuEraser python: ${DIFF_PY}"
else
  fail "DiffuEraser env python not found; set DIFFUERASER_ENV=/path/to/env"
fi
if [[ -n "${VIDEO_PY}" ]]; then
  ok "VideoDPO python: ${VIDEO_PY}"
else
  warn "VideoDPO env python not found; set VIDEODPO_ENV=/path/to/env if running official VideoDPO/VBench"
fi

if [[ -n "${DIFF_PY}" ]]; then
  run_python_imports "diffueraser" "${DIFF_PY}" torch accelerate transformers diffusers decord || fail "DiffuEraser import check failed"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${DIFF_PY}" - <<'PY' || fail "DiffuEraser torch cuda check failed"
import torch
print(f"[OK] torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
PY
fi
if [[ -n "${VIDEO_PY}" ]]; then
  run_python_imports "videodpo" "${VIDEO_PY}" torch pytorch_lightning omegaconf kornia open_clip transformers huggingface_hub diffusers fairscale decord av || fail "VideoDPO import check failed"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONPATH="${VIDEODPO_REPO}:${PYTHONPATH:-}" "${VIDEO_PY}" - <<'PY' || fail "VideoDPO torch/lvdm check failed"
import torch
import lvdm.modules.encoders.condition
import lvdm.models.ddpm3d
print(f"[OK] torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
print("[OK] lvdm imports")
PY
fi

section "Weights"
check_dir "${WEIGHTS_DIR}" "weights root"
check_dir "${WEIGHTS_DIR}/stable-diffusion-v1-5" "stable-diffusion-v1-5"
check_dir "${WEIGHTS_DIR}/stable-diffusion-v1-5/tokenizer" "stable-diffusion-v1-5 tokenizer"
check_dir "${WEIGHTS_DIR}/sd-vae-ft-mse" "sd-vae-ft-mse"
check_dir "${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000" "DiffuEraser converted_weights_step48000"
check_file "${VIDEODPO_REPO}/checkpoints/vc2/model.ckpt" "VC2 base model.ckpt"
check_file "${VIDEODPO_REPO}/checkpoints/vc2/ref_model.ckpt" "VC2 ref_model.ckpt"

section "Full-Mask Required Weight Files"
SD15_DIR="${WEIGHTS_DIR}/stable-diffusion-v1-5"
VAE_DIR="${WEIGHTS_DIR}/sd-vae-ft-mse"
DIFFUERASER_REF_DIR="${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000"
check_file "${SD15_DIR}/model_index.json" "SD1.5 model_index.json"
check_file "${SD15_DIR}/tokenizer/vocab.json" "SD1.5 tokenizer vocab.json"
check_file "${SD15_DIR}/tokenizer/merges.txt" "SD1.5 tokenizer merges.txt"
check_file "${SD15_DIR}/tokenizer/tokenizer_config.json" "SD1.5 tokenizer config"
check_file "${SD15_DIR}/scheduler/scheduler_config.json" "SD1.5 scheduler config"
check_file "${SD15_DIR}/text_encoder/config.json" "SD1.5 text_encoder config"
check_file_any "SD1.5 text_encoder weights" \
  "${SD15_DIR}/text_encoder/model.safetensors" \
  "${SD15_DIR}/text_encoder/pytorch_model.bin" \
  "${SD15_DIR}/text_encoder/model.fp16.safetensors" \
  "${SD15_DIR}/text_encoder/pytorch_model.fp16.bin"
check_file "${SD15_DIR}/unet/config.json" "SD1.5 unet config"
check_file_any "SD1.5 unet weights" \
  "${SD15_DIR}/unet/diffusion_pytorch_model.safetensors" \
  "${SD15_DIR}/unet/diffusion_pytorch_model.bin" \
  "${SD15_DIR}/unet/diffusion_pytorch_model.fp16.safetensors" \
  "${SD15_DIR}/unet/diffusion_pytorch_model.fp16.bin"
check_file "${VAE_DIR}/config.json" "sd-vae-ft-mse config"
check_file_any "sd-vae-ft-mse weights" \
  "${VAE_DIR}/diffusion_pytorch_model.safetensors" \
  "${VAE_DIR}/diffusion_pytorch_model.bin"
check_file "${DIFFUERASER_REF_DIR}/unet_main/config.json" "DiffuEraser ref unet_main config"
check_file_any "DiffuEraser ref unet_main weights" \
  "${DIFFUERASER_REF_DIR}/unet_main/diffusion_pytorch_model.safetensors" \
  "${DIFFUERASER_REF_DIR}/unet_main/diffusion_pytorch_model.bin"
check_file "${DIFFUERASER_REF_DIR}/brushnet/config.json" "DiffuEraser ref brushnet config"
check_file_any "DiffuEraser ref brushnet weights" \
  "${DIFFUERASER_REF_DIR}/brushnet/diffusion_pytorch_model.safetensors" \
  "${DIFFUERASER_REF_DIR}/brushnet/diffusion_pytorch_model.bin"

section "VideoDPO VC2 Data"
RESOLVED_DPO_DATA_ROOT=""
if command -v python3 >/dev/null 2>&1; then
  resolved="$(resolve_vc2_yaml python3 2>/tmp/pai_resolve_data.$$ || true)"
  if [[ -n "${resolved}" ]]; then
    RESOLVED_DPO_DATA_ROOT="${resolved}"
    ok "resolved DPO data root/yaml: ${RESOLVED_DPO_DATA_ROOT}"
  else
    fail "could not resolve VideoDPO VC2 train_data yaml or dataset root"
    cat /tmp/pai_resolve_data.$$ 2>/dev/null || true
  fi
  rm -f /tmp/pai_resolve_data.$$
else
  warn "python3 not found; cannot resolve VideoDPO data"
fi

if command -v python3 >/dev/null 2>&1 && [[ -n "${RESOLVED_DPO_DATA_ROOT}" ]]; then
  DPO_DATA_ROOT="${RESOLVED_DPO_DATA_ROOT}" python3 - <<'PY' || fail "VideoDPO data validation failed"
from pathlib import Path
import json
import os

path = Path(os.environ["DPO_DATA_ROOT"]).expanduser()
roots = []
if path.is_dir():
    roots = [path]
else:
    try:
        import yaml
        cfg = yaml.safe_load(path.read_text()) or {}
        for item in cfg.get("META", []):
            root = Path(str(item)).expanduser()
            if not root.is_absolute():
                root = path.parent / root
            roots.append(root)
    except Exception as exc:
        raise SystemExit(f"failed to read yaml {path}: {exc}")

if not roots:
    raise SystemExit("no META roots")
for root in roots:
    metadata_path = root / "metadata.json"
    pair_path = root / "pair.json"
    if not metadata_path.is_file() or not pair_path.is_file():
        raise SystemExit(f"missing metadata/pair under {root}")
    metadata = json.load(metadata_path.open())
    pairs = json.load(pair_path.open())
    print(f"[OK] data root={root}")
    print(f"[OK] metadata={len(metadata)} pair={len(pairs)}")
    if metadata:
        clip = Path(metadata[0]["basic"]["clip_path"])
        if not clip.is_absolute():
            clip = root / clip
        print(f"[OK] first_clip={clip} exists={clip.is_file()}")
        if not clip.is_file():
            raise SystemExit(f"first clip missing: {clip}")
PY
else
  warn "skip VideoDPO data validation because no valid train_data yaml/root was resolved"
fi

section "Full-Mask Dataset Smoke"
if [[ "${RUN_DATASET_SMOKE}" == "1" ]]; then
  if [[ -n "${DIFF_PY}" && -e "${RESOLVED_DPO_DATA_ROOT}" && -d "${WEIGHTS_DIR}/stable-diffusion-v1-5" ]]; then
    CUDA_VISIBLE_DEVICES="${SMOKE_GPU}" \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    "${DIFF_PY}" "${PROJECT_ROOT}/tools/smoke_videodpo_fullmask_dataset.py" \
      --dpo_data_root "${RESOLVED_DPO_DATA_ROOT}" \
      --base_model_name_or_path "${WEIGHTS_DIR}/stable-diffusion-v1-5" \
      --resolution 512 \
      --train_height 320 \
      --train_width 512 \
      --nframes 16 \
      --videodpo_frame_stride 1 \
      --videodpo_full_mask_value 0.0 \
      --index 0 || fail "full-mask dataset smoke failed"
  else
    warn "skip full-mask dataset smoke because env/data/base model is missing"
  fi
else
  warn "full-mask dataset smoke disabled by RUN_DATASET_SMOKE=${RUN_DATASET_SMOKE}"
fi

section "Optional Training Smokes"
echo "Full-mask train smoke command:"
cat <<EOF
CUDA_VISIBLE_DEVICES=${SMOKE_GPU} \\
DIFFUERASER_ENV=${DIFFUERASER_ENV:-${PROJECT_HOME}/conda_envs/diffueraser} \\
CONDA_ENV=${DIFFUERASER_ENV:-${PROJECT_HOME}/conda_envs/diffueraser} \\
PROJECT_ROOT=${PROJECT_ROOT} \\
PROJECT_HOME=${PROJECT_HOME} \\
PROJECT_DEV=${PROJECT_DEV} \\
PROJECT_DATA=${PROJECT_DATA} \\
DATA=${DATA} \\
WEIGHTS_DIR=${WEIGHTS_DIR} \\
DPO_DATA_ROOT=${RESOLVED_DPO_DATA_ROOT} \\
RUN_NAME=pai-gpu${SMOKE_GPU}-videodpo-fullmask-diffueraser-smoke-\$(date +%Y%m%d_%H%M%S) \\
NUM_GPUS=1 \\
MAX_STEPS=2 \\
CKPT_STEPS=999999 \\
VAL_STEPS=999999 \\
LOGGING_STEPS=1 \\
REPORT_TO=none \\
bash DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage1.sbatch
EOF

if [[ "${RUN_FULLMASK_TRAIN_SMOKE}" == "1" ]]; then
  if [[ -z "${DIFF_PY}" ]]; then
    fail "cannot run full-mask train smoke without DiffuEraser env"
  elif [[ -z "${RESOLVED_DPO_DATA_ROOT}" || ! -e "${RESOLVED_DPO_DATA_ROOT}" ]]; then
    fail "cannot run full-mask train smoke without resolved VideoDPO VC2 data"
  else
    CUDA_VISIBLE_DEVICES="${SMOKE_GPU}" \
    CONDA_ENV="$(cd "$(dirname "${DIFF_PY}")/.." && pwd)" \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    PROJECT_HOME="${PROJECT_HOME}" \
    PROJECT_DEV="${PROJECT_DEV}" \
    PROJECT_DATA="${PROJECT_DATA}" \
    DATA="${DATA}" \
    WEIGHTS_DIR="${WEIGHTS_DIR}" \
    DPO_DATA_ROOT="${RESOLVED_DPO_DATA_ROOT}" \
    RUN_NAME="pai-gpu${SMOKE_GPU}-videodpo-fullmask-diffueraser-smoke-$(date +%Y%m%d_%H%M%S)" \
    NUM_GPUS=1 \
    MAX_STEPS=2 \
    CKPT_STEPS=999999 \
    VAL_STEPS=999999 \
    LOGGING_STEPS=1 \
    REPORT_TO=none \
    bash "${PROJECT_ROOT}/DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage1.sbatch" || fail "full-mask train smoke failed"
  fi
else
  warn "full-mask train smoke not run; set RUN_FULLMASK_TRAIN_SMOKE=1 after env/data/weights pass"
fi

if [[ "${RUN_VIDEODPO_TRAIN_SMOKE}" == "1" ]]; then
  if [[ -z "${VIDEO_PY}" ]]; then
    fail "cannot run official VideoDPO smoke without VideoDPO env"
  elif [[ -z "${RESOLVED_DPO_DATA_ROOT}" || ! -e "${RESOLVED_DPO_DATA_ROOT}" ]]; then
    fail "cannot run official VideoDPO smoke without resolved VideoDPO VC2 data"
  else
    SMOKE=1 \
    CUDA_VISIBLE_DEVICES="${SMOKE_GPU}" \
    NUM_GPUS=1 \
    DEVICE_LIST="${SMOKE_GPU}" \
    ENABLE_WANDB=0 \
    EARLY_WANDB=0 \
    WANDB_START_EVENT=0 \
    VIDEODPO_REPO="${VIDEODPO_REPO}" \
    CONDA_ENV="$(cd "$(dirname "${VIDEO_PY}")/.." && pwd)" \
    VC2_DATA_YAML="${RESOLVED_DPO_DATA_ROOT}" \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    PROJECT_HOME="${PROJECT_HOME}" \
    PROJECT_DEV="${PROJECT_DEV}" \
    PROJECT_DATA="${PROJECT_DATA}" \
    RUN_NAME="pai-vc2-dpo-official-smoke-gpu${SMOKE_GPU}-$(date +%Y%m%d_%H%M%S)" \
    bash "${PROJECT_ROOT}/DPO_finetune/scripts/h20_videodpo_vc2_train.sh" || fail "official VideoDPO train smoke failed"
  fi
else
  warn "official VideoDPO train smoke not run; set RUN_VIDEODPO_TRAIN_SMOKE=1 if needed"
fi

section "Summary"
echo "fails=${FAILS} warns=${WARNS}"
if [[ "${FAILS}" -gt 0 ]]; then
  echo "[SUMMARY] health check failed"
  exit 1
fi
if [[ "${STRICT}" == "1" && "${WARNS}" -gt 0 ]]; then
  echo "[SUMMARY] strict mode failed because warnings were found"
  exit 1
fi
echo "[SUMMARY] health check passed"
