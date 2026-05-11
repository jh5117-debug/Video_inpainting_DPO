#!/usr/bin/env bash
# Static SC health check for VideoDPO reproduction and DiffuEraser bridge.
# This script intentionally does not run Python, training, inference, VBench, or
# any GPU command.  It is safe for a compute node or login-side static audit.

set -uo pipefail

ERRORS=0
WARNINGS=0

section() {
  printf '\n========== %s ==========\n' "$1"
}

ok() {
  printf '[OK] %s\n' "$1"
}

warn() {
  WARNINGS=$((WARNINGS + 1))
  printf '[WARN] %s\n' "$1"
}

fail() {
  ERRORS=$((ERRORS + 1))
  printf '[FAIL] %s\n' "$1"
}

check_dir() {
  local path="$1"
  local label="$2"
  if [[ -d "$path" ]]; then
    ok "$label: $path"
  else
    fail "$label missing: $path"
  fi
}

check_file() {
  local path="$1"
  local label="$2"
  if [[ -f "$path" ]]; then
    ok "$label: $path"
    ls -lh "$path" 2>/dev/null || true
  else
    fail "$label missing: $path"
  fi
}

check_exe_or_file() {
  local path="$1"
  local label="$2"
  if [[ -f "$path" ]]; then
    ok "$label: $path"
  else
    fail "$label missing: $path"
  fi
}

first_meta_from_yaml() {
  local yaml_path="$1"
  sed -n 's/^[[:space:]]*-[[:space:]]*//p' "$yaml_path" \
    | head -n 1 \
    | sed 's/[[:space:]]*$//' \
    | sed 's/^["'\'']//; s/["'\'']$//'
}

resolve_meta_path() {
  local yaml_path="$1"
  local meta_path="$2"
  local yaml_dir
  yaml_dir="$(cd "$(dirname "$yaml_path")" && pwd)"

  if [[ "$meta_path" = /* ]]; then
    printf '%s\n' "$meta_path"
    return 0
  fi
  if [[ -d "${VIDEODPO_DATA_BASE}/${meta_path}" ]]; then
    printf '%s\n' "${VIDEODPO_DATA_BASE}/${meta_path}"
    return 0
  fi
  if [[ -d "${VIDEODPO_REPO}/${meta_path}" ]]; then
    printf '%s\n' "${VIDEODPO_REPO}/${meta_path}"
    return 0
  fi
  printf '%s\n' "${yaml_dir}/${meta_path}"
}

count_json_key() {
  local key="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    grep -o "\"${key}\"" "$path" | wc -l | tr -d ' '
  else
    printf '0'
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="${PROJECT_NAME:-Video_inpainting_DPO}"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${PROJECT_DEV:-}" ]]; then
    PROJECT_ROOT="${PROJECT_DEV}/${PROJECT_NAME}"
  else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" 2>/dev/null && pwd || printf '%s' "$PROJECT_ROOT")"

PROJECT_HOME="${PROJECT_HOME:-/sc-projects/sc-proj-cc09-repair/hongyou}"
PROJECT_DEV="${PROJECT_DEV:-${PROJECT_HOME}/dev}"
PROJECT_DATA="${PROJECT_DATA:-${PROJECT_DEV}/data}"
VIDEODPO_REPO="${VIDEODPO_REPO:-${PROJECT_ROOT}/external/VideoDPO}"
VIDEODPO_DATA_BASE="${VIDEODPO_DATA_BASE:-${PROJECT_DATA}/VideoDPO}"
VBENCH_ROOT="${VBENCH_ROOT:-${PROJECT_ROOT}/external/VBench}"
VC2_DATA_YAML="${VC2_DATA_YAML:-${PROJECT_DATA}/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml}"
VC2_DATASET_ROOT="${VC2_DATASET_ROOT:-${PROJECT_DATA}/VideoDPO/data/vidpro-vc2-dpo-dataset}"
PROMPTS_FILE="${PROMPTS_FILE:-${VIDEODPO_REPO}/prompts/vbench_standard_prompts.txt}"
CONDA_ENV="${CONDA_ENV:-${VIDEODPO_CONDA_ENV:-videodpo}}"
DIFFUERASER_WEIGHTS_DIR="${WEIGHTS_DIR:-${PROJECT_DATA}/Video_inpainting_DPO/weights}"
DIFFUERASER_REF="${REF_MODEL_PATH:-${DIFFUERASER_WEIGHTS_DIR}/diffuEraser/converted_weights_step48000}"

section "Runtime"
date
hostname
printf 'USER=%s\n' "${USER:-unknown}"
printf 'PWD=%s\n' "$PWD"
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-none}"
printf 'SLURM_JOB_PARTITION=%s\n' "${SLURM_JOB_PARTITION:-none}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  warn "CUDA_VISIBLE_DEVICES is set (${CUDA_VISIBLE_DEVICES}); this health check does not need GPU."
else
  ok "CUDA_VISIBLE_DEVICES is unset; no GPU requested by this script."
fi
if [[ -n "${SLURM_JOB_PARTITION:-}" && "${SLURM_JOB_PARTITION}" != *compute* ]]; then
  warn "This static check should run on a compute/non-GPU partition if submitted through Slurm."
fi

section "Project Paths"
printf 'PROJECT_HOME=%s\n' "$PROJECT_HOME"
printf 'PROJECT_DEV=%s\n' "$PROJECT_DEV"
printf 'PROJECT_DATA=%s\n' "$PROJECT_DATA"
printf 'PROJECT_ROOT=%s\n' "$PROJECT_ROOT"
printf 'VIDEODPO_REPO=%s\n' "$VIDEODPO_REPO"
printf 'VIDEODPO_DATA_BASE=%s\n' "$VIDEODPO_DATA_BASE"
printf 'VBENCH_ROOT=%s\n' "$VBENCH_ROOT"
check_dir "$PROJECT_DEV" "PROJECT_DEV"
check_dir "$PROJECT_DATA" "PROJECT_DATA"
check_dir "$PROJECT_ROOT" "Video_inpainting_DPO repo"
check_dir "$VIDEODPO_REPO" "VideoDPO repo"
check_dir "$VBENCH_ROOT" "VBench repo"

section "Git State"
if command -v git >/dev/null 2>&1 && [[ -d "${PROJECT_ROOT}/.git" ]]; then
  git -C "$PROJECT_ROOT" status -sb || warn "git status failed for $PROJECT_ROOT"
  git -C "$PROJECT_ROOT" log -1 --oneline || warn "git log failed for $PROJECT_ROOT"
else
  warn "git not available or PROJECT_ROOT is not a git repo"
fi
section "Submodules"
check_file "${PROJECT_ROOT}/.gitmodules" ".gitmodules"
if command -v git >/dev/null 2>&1 && [[ -d "${PROJECT_ROOT}/.git" ]]; then
  SUBMODULE_STATUS="$(git -C "$PROJECT_ROOT" submodule status --recursive 2>/dev/null || true)"
  if [[ -n "$SUBMODULE_STATUS" ]]; then
    printf '%s\n' "$SUBMODULE_STATUS"
    if printf '%s\n' "$SUBMODULE_STATUS" | grep -q '^-'; then
      fail "At least one submodule is not initialized. Run: git submodule update --init --recursive"
    fi
  else
    fail "No submodule status output; expected external/VideoDPO and external/VBench"
  fi
else
  warn "Cannot inspect submodules because git or PROJECT_ROOT/.git is unavailable"
fi
check_file "${VIDEODPO_REPO}/scripts/train.py" "VideoDPO submodule train.py"
check_file "${VBENCH_ROOT}/evaluate.py" "VBench submodule evaluate.py"

section "Dependency Git State"
if command -v git >/dev/null 2>&1 && [[ -e "${VIDEODPO_REPO}/.git" ]]; then
  git -C "$VIDEODPO_REPO" status -sb || warn "git status failed for $VIDEODPO_REPO"
  git -C "$VIDEODPO_REPO" log -1 --oneline || warn "git log failed for $VIDEODPO_REPO"
else
  warn "VideoDPO repo git metadata not available"
fi
if command -v git >/dev/null 2>&1 && [[ -e "${VBENCH_ROOT}/.git" ]]; then
  git -C "$VBENCH_ROOT" status -sb || warn "git status failed for $VBENCH_ROOT"
  git -C "$VBENCH_ROOT" log -1 --oneline || warn "git log failed for $VBENCH_ROOT"
else
  warn "VBench repo git metadata not available"
fi

section "VideoDPO VC2 Dataset"
check_file "$VC2_DATA_YAML" "absolute VC2 train yaml"
if [[ -f "$VC2_DATA_YAML" ]]; then
  printf -- '--- %s ---\n' "$VC2_DATA_YAML"
  sed -n '1,20p' "$VC2_DATA_YAML"
  META_PATH="$(first_meta_from_yaml "$VC2_DATA_YAML")"
  if [[ -z "$META_PATH" ]]; then
    fail "Could not parse first META entry from $VC2_DATA_YAML"
  else
    printf 'first META entry=%s\n' "$META_PATH"
    if [[ "$META_PATH" != /* ]]; then
      warn "META entry is relative; SC training is safer with absolute train_data.absolute.yaml"
    fi
    RESOLVED_META="$(resolve_meta_path "$VC2_DATA_YAML" "$META_PATH")"
    printf 'resolved META root=%s\n' "$RESOLVED_META"
    check_dir "$RESOLVED_META" "resolved VideoDPO dataset root"
    check_file "${RESOLVED_META}/metadata.json" "metadata.json"
    check_file "${RESOLVED_META}/pair.json" "pair.json"
    if [[ -f "${RESOLVED_META}/metadata.json" ]]; then
      META_COUNT="$(count_json_key basic "${RESOLVED_META}/metadata.json")"
      printf 'metadata basic-key count=%s\n' "$META_COUNT"
      if [[ "$META_COUNT" -le 0 ]]; then
        warn "metadata.json basic-key count is zero; inspect JSON format manually"
      fi
      FIRST_CLIP="$(grep -m 1 -o '"clip_path"[[:space:]]*:[[:space:]]*"[^"]*"' "${RESOLVED_META}/metadata.json" | sed 's/.*"clip_path"[[:space:]]*:[[:space:]]*"//; s/"$//' || true)"
      if [[ -n "$FIRST_CLIP" ]]; then
        if [[ "$FIRST_CLIP" = /* ]]; then
          FIRST_CLIP_PATH="$FIRST_CLIP"
        else
          FIRST_CLIP_PATH="${RESOLVED_META}/${FIRST_CLIP}"
        fi
        if [[ -f "$FIRST_CLIP_PATH" ]]; then
          ok "first clip exists: $FIRST_CLIP_PATH"
          ls -lh "$FIRST_CLIP_PATH" 2>/dev/null || true
        else
          fail "first clip missing: $FIRST_CLIP_PATH"
        fi
      else
        warn "Could not parse first clip_path from metadata.json"
      fi
    fi
    if [[ -f "${RESOLVED_META}/pair.json" ]]; then
      PAIR_COUNT="$(count_json_key video1 "${RESOLVED_META}/pair.json")"
      printf 'pair video1-key count=%s\n' "$PAIR_COUNT"
      if [[ "$PAIR_COUNT" -le 0 ]]; then
        warn "pair.json video1-key count is zero; inspect JSON format manually"
      fi
    fi
  fi
else
  warn "Expected prepared yaml missing. Default dataset root was: $VC2_DATASET_ROOT"
fi

section "VC2 Base And Ref Checkpoints"
check_file "${VIDEODPO_REPO}/checkpoints/vc2/model.ckpt" "VC2 base model.ckpt"
check_file "${VIDEODPO_REPO}/checkpoints/vc2/ref_model.ckpt" "VC2 ref_model.ckpt"
if [[ -d "${VIDEODPO_REPO}/checkpoints/vc2" ]]; then
  du -sh "${VIDEODPO_REPO}/checkpoints/vc2" 2>/dev/null || true
fi

section "VideoDPO Environment Static Checks"
if command -v conda >/dev/null 2>&1; then
  ok "conda command found"
  conda env list 2>/dev/null | grep -E "(^|[[:space:]])${CONDA_ENV}([[:space:]]|$)" \
    && ok "conda env listed: ${CONDA_ENV}" \
    || warn "conda env not listed by name: ${CONDA_ENV}"
else
  warn "conda command not found in current shell"
fi
check_file "${VIDEODPO_REPO}/requirements.txt" "VideoDPO requirements.txt"
check_file "${VIDEODPO_REPO}/scripts/train.py" "VideoDPO train.py"
check_file "${VIDEODPO_REPO}/configs/vc2_dpo/config.yaml" "official VC2-DPO config"
check_file "${VIDEODPO_REPO}/data/video_data.py" "official VideoDPO dataloader"

section "Official VC2-DPO Hyperparameter Lines"
VC2_CONFIG="${VIDEODPO_REPO}/configs/vc2_dpo/config.yaml"
if [[ -f "$VC2_CONFIG" ]]; then
  grep -nE 'base_learning_rate|beta_dpo|batch_size:|data_root:|resolution:|video_length:|accumulate_grad_batches|max_epochs|every_n_train_steps|num_workers:' "$VC2_CONFIG" || true
else
  fail "Cannot inspect official config because it is missing"
fi

section "VBench Static Checks"
check_exe_or_file "${VBENCH_ROOT}/evaluate.py" "VBench evaluate.py"
check_file "${VBENCH_ROOT}/vbench/VBench_full_info.json" "VBench full info json"
check_file "$PROMPTS_FILE" "VideoDPO VBench prompt file"
if [[ -f "$PROMPTS_FILE" ]]; then
  printf 'prompt count: '
  wc -l "$PROMPTS_FILE"
  printf 'first prompts:\n'
  head -n 5 "$PROMPTS_FILE"
fi
if [[ -d "${HOME}/.cache/vbench" ]]; then
  ok "VBench cache exists: ${HOME}/.cache/vbench"
  du -sh "${HOME}/.cache/vbench" 2>/dev/null || true
else
  warn "VBench cache not found at ${HOME}/.cache/vbench; metric weights may download on first evaluation"
fi

section "DiffuEraser Bridge Static Checks"
printf 'DIFFUERASER_WEIGHTS_DIR=%s\n' "$DIFFUERASER_WEIGHTS_DIR"
printf 'DIFFUERASER_REF=%s\n' "$DIFFUERASER_REF"
if [[ ! -d "$DIFFUERASER_REF" && -d "/home/hj/Video_inpainting_DPO/weights/diffuEraser/converted_weights_step48000" ]]; then
  warn "Default DiffuEraser ref missing, but /home/hj/Video_inpainting_DPO/weights/diffuEraser/converted_weights_step48000 exists. Use WEIGHTS_DIR=/home/hj/Video_inpainting_DPO/weights on SC."
fi
check_dir "$DIFFUERASER_REF" "DiffuEraser converted_weights_step48000"
check_file "${DIFFUERASER_REF}/unet_main/config.json" "DiffuEraser unet_main config"
check_file "${DIFFUERASER_REF}/brushnet/config.json" "DiffuEraser brushnet config"

section "Repository Script Syntax"
for script in \
  "${PROJECT_ROOT}/DPO_finetune/scripts/sc_prepare_videodpo_vc2_assets.sbatch" \
  "${PROJECT_ROOT}/DPO_finetune/scripts/sc_videodpo_vc2_train.sbatch" \
  "${PROJECT_ROOT}/DPO_finetune/scripts/sc_videodpo_vc2_vbench.sbatch" \
  "${PROJECT_ROOT}/DPO_finetune/scripts/sc_videodpo_vc2_checkpoint_sweep.sbatch" \
  "${PROJECT_ROOT}/DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage1.sbatch" \
  "${PROJECT_ROOT}/DPO_finetune/scripts/sc_videodpo_fullmask_dataset_smoke.sbatch" \
  "${PROJECT_ROOT}/DPO_finetune/scripts/videodpo_env_smoke_and_export.sh" \
  "${PROJECT_ROOT}/DPO_finetune/scripts/sc_prepare_videodpo_vc2_checkpoints.sh" \
  "${PROJECT_ROOT}/DPO_finetune/scripts/sc_videodpo_pull_submodules_and_health_check.sh" \
  "${PROJECT_ROOT}/DPO_finetune/scripts/sc_videodpo_health_check.sh" \
  "${PROJECT_ROOT}/DPO_finetune/scripts/03_dpo_stage1.sbatch" \
  "${PROJECT_ROOT}/DPO_finetune/scripts/03_dpo_stage2.sbatch" \
  "${PROJECT_ROOT}/training/dpo/scripts/03_dpo_stage1.sbatch" \
  "${PROJECT_ROOT}/training/dpo/scripts/03_dpo_stage2.sbatch"
do
  if [[ -f "$script" ]]; then
    if bash -n "$script"; then
      ok "bash -n: $script"
    else
      fail "bash syntax failed: $script"
    fi
  else
    warn "script not found for syntax check: $script"
  fi
done

section "Summary"
printf 'errors=%d warnings=%d\n' "$ERRORS" "$WARNINGS"
if [[ "$ERRORS" -gt 0 ]]; then
  printf '[RESULT] FAIL: fix missing required assets before training/evaluation.\n'
  exit 1
fi
if [[ "$WARNINGS" -gt 0 ]]; then
  printf '[RESULT] PASS WITH WARNINGS: review warnings before long jobs.\n'
  exit 0
fi
printf '[RESULT] PASS: static health check found required assets.\n'
