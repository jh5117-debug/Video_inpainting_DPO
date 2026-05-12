#!/usr/bin/env bash
# SC health check for VideoDPO reproduction and DiffuEraser bridge.
# This script never runs training, inference, VBench, or any GPU command.  By
# default it does run a lightweight Python import preflight so missing conda
# packages fail here instead of after a queued GPU job starts.

set -uo pipefail

if [[ "${VERBOSE:-0}" == "1" ]]; then
  QUIET_OK=0
else
  QUIET_OK="${QUIET_OK:-1}"
fi

if [[ "${SC_HEALTH_FILTERED:-0}" != "1" && "${QUIET_OK}" == "1" ]]; then
  export SC_HEALTH_FILTERED=1
  set +e
  "$0" "$@" 2>&1 | awk '
    /^\[FAIL\]/ ||
    /^\[WARN\]/ ||
    /^\[DIAG\]/ ||
    /^========== Summary ==========/ ||
    /^errors=/ ||
    /^failures:/ ||
    /^warnings:/ ||
    /^  - / ||
    /^dataset_root=/ ||
    /^metadata_clip_path=/ ||
    /^metadata_clip_exists=/ ||
    /^real_local_candidate=/ ||
    /^repo_commit=/ ||
    /^\[RESULT\]/ { print }
  '
  exit "${PIPESTATUS[0]}"
fi

ERRORS=0
WARNINGS=0
FAIL_MESSAGES=()
WARN_MESSAGES=()

section() {
  printf '\n========== %s ==========\n' "$1"
}

ok() {
  printf '[OK] %s\n' "$1"
}

warn() {
  WARNINGS=$((WARNINGS + 1))
  WARN_MESSAGES+=("$1")
  printf '[WARN] %s\n' "$1"
}

fail() {
  ERRORS=$((ERRORS + 1))
  FAIL_MESSAGES+=("$1")
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

first_json_string_value() {
  local key="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    grep -o "\"${key}\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" "$path" \
      | head -n 1 \
      | sed "s/.*\"${key}\"[[:space:]]*:[[:space:]]*\"//; s/\"$//"
  fi
}

diag_path_status() {
  local label="$1"
  local path="$2"
  local parent
  parent="$(dirname "$path")"
  printf '[DIAG] %s=%s\n' "$label" "$path"
  printf '[DIAG] %s_exists=%s is_file=%s is_dir=%s is_symlink=%s\n' \
    "$label" \
    "$([[ -e "$path" ]] && echo yes || echo no)" \
    "$([[ -f "$path" ]] && echo yes || echo no)" \
    "$([[ -d "$path" ]] && echo yes || echo no)" \
    "$([[ -L "$path" ]] && echo yes || echo no)"
  printf '[DIAG] %s_parent=%s parent_exists=%s parent_is_dir=%s\n' \
    "$label" "$parent" \
    "$([[ -e "$parent" ]] && echo yes || echo no)" \
    "$([[ -d "$parent" ]] && echo yes || echo no)"
  ls -ld "$path" 2>/dev/null | sed "s/^/[DIAG] ls_${label}=/" || true
  ls -ld "$parent" 2>/dev/null | sed "s/^/[DIAG] ls_${label}_parent=/" || true
  readlink -f "$path" 2>/dev/null | sed "s/^/[DIAG] readlink_${label}=/" || true
}

find_limited() {
  local root="$1"
  shift
  local timeout_s="${HEALTH_FIND_TIMEOUT:-20}"
  local limit="${HEALTH_FIND_LIMIT:-20}"
  if command -v timeout >/dev/null 2>&1; then
    timeout "$timeout_s" find "$root" "$@" 2>/dev/null | head -n "$limit" || true
  else
    find "$root" "$@" 2>/dev/null | head -n "$limit" || true
  fi
}

shown_count() {
  sed '/^$/d' | wc -l | tr -d ' '
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
CHECK_ENV_IMPORTS="${CHECK_ENV_IMPORTS:-1}"
REQUIRE_WANDB="${REQUIRE_WANDB:-1}"
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
  printf 'repo_commit=%s\n' "$(git -C "$PROJECT_ROOT" log -1 --oneline 2>/dev/null || true)"
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
      FIRST_CLIP="$(first_json_string_value clip_path "${RESOLVED_META}/metadata.json" || true)"
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
          CLIP_TRIMMED="${FIRST_CLIP%/}"
          CLIP_BASENAME="${CLIP_TRIMMED##*/}"
          CLIP_PARENT_PATH="${CLIP_TRIMMED%/*}"
          CLIP_PARENT="${CLIP_PARENT_PATH##*/}"
          CLIP_SUFFIX="${CLIP_PARENT}/${CLIP_BASENAME}"
          EXTRACTED_ROOT="${VC2_DATASET_ROOT}/_extracted"
          printf '[DIAG] dataset_root=%s\n' "$RESOLVED_META"
          printf '[DIAG] metadata_clip_path=%s\n' "$FIRST_CLIP"
          printf '[DIAG] resolved_metadata_clip=%s\n' "$FIRST_CLIP_PATH"
          printf '[DIAG] metadata_clip_exists=%s\n' "$([ -f "$FIRST_CLIP_PATH" ] && echo yes || echo no)"
          diag_path_status first_clip "$FIRST_CLIP_PATH"
          printf '[DIAG] parsed_clip_parent=%s parsed_clip_basename=%s\n' "$CLIP_PARENT" "$CLIP_BASENAME"
          printf '[DIAG] extracted_root=%s exists=%s\n' "$EXTRACTED_ROOT" "$([ -d "$EXTRACTED_ROOT" ] && echo yes || echo no)"
          if [[ -d "$EXTRACTED_ROOT" ]]; then
            printf '[DIAG] bounded_find_timeout_seconds=%s limit=%s\n' "${HEALTH_FIND_TIMEOUT:-20}" "${HEALTH_FIND_LIMIT:-20}"
            VIDEO_SAMPLE="$(find_limited "$EXTRACTED_ROOT" -type f \( -name '*.mp4' -o -name '*.webm' -o -name '*.avi' -o -name '*.gif' \) -print)"
            SUFFIX_SAMPLE="$(find_limited "$EXTRACTED_ROOT" -type f -path "*/${CLIP_SUFFIX}" -print)"
            BASENAME_SAMPLE="$(find_limited "$EXTRACTED_ROOT" -type f -name "$CLIP_BASENAME" -print)"
            PARENT_DIR_SAMPLE="$(find_limited "$EXTRACTED_ROOT" -type d -name "$CLIP_PARENT" -print)"
            VIDEO_SHOWN="$(printf '%s\n' "$VIDEO_SAMPLE" | shown_count)"
            SUFFIX_SHOWN="$(printf '%s\n' "$SUFFIX_SAMPLE" | shown_count)"
            BASENAME_SHOWN="$(printf '%s\n' "$BASENAME_SAMPLE" | shown_count)"
            PARENT_DIR_SHOWN="$(printf '%s\n' "$PARENT_DIR_SAMPLE" | shown_count)"
            printf '[DIAG] video_sample_shown=%s\n' "$VIDEO_SHOWN"
            printf '%s\n' "$VIDEO_SAMPLE" | sed '/^$/d' | sed 's/^/[DIAG] video_sample=/'
            printf '[DIAG] search_suffix=%s shown=%s\n' "$CLIP_SUFFIX" "$SUFFIX_SHOWN"
            printf '%s\n' "$SUFFIX_SAMPLE" | sed '/^$/d' | sed 's/^/[DIAG] candidate_by_suffix=/'
            printf '[DIAG] search_basename=%s shown=%s\n' "$CLIP_BASENAME" "$BASENAME_SHOWN"
            printf '%s\n' "$BASENAME_SAMPLE" | sed '/^$/d' | sed 's/^/[DIAG] candidate_by_basename=/'
            printf '[DIAG] search_parent_dir=%s shown=%s\n' "$CLIP_PARENT" "$PARENT_DIR_SHOWN"
            printf '%s\n' "$PARENT_DIR_SAMPLE" | sed '/^$/d' | sed 's/^/[DIAG] candidate_parent_dir=/'
            if [[ "$SUFFIX_SHOWN" -gt 0 || "$BASENAME_SHOWN" -gt 0 ]]; then
              printf '[DIAG] diagnosis=metadata clip_path points to the wrong location; rerun sc_prepare_videodpo_vc2_assets.sbatch after pulling the latest repo.\n'
              printf '[DIAG] repair_command=CONDA_ENV=diffueraser DOWNLOAD_DATASET=0 sbatch --export=ALL DPO_finetune/scripts/sc_prepare_videodpo_vc2_assets.sbatch\n'
            else
              printf '[DIAG] diagnosis=no local video candidate was found in the bounded scan under _extracted; dataset extraction/download may be incomplete, or increase HEALTH_FIND_TIMEOUT.\n'
              printf '[DIAG] repair_command=CONDA_ENV=diffueraser sbatch --export=ALL DPO_finetune/scripts/sc_prepare_videodpo_vc2_assets.sbatch\n'
            fi
          fi
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
if [[ "${CHECK_ENV_IMPORTS}" == "1" ]]; then
  if command -v conda >/dev/null 2>&1; then
    ENV_IMPORT_OUTPUT="$(
      VIDEODPO_REPO="${VIDEODPO_REPO}" REQUIRE_WANDB="${REQUIRE_WANDB}" \
      conda run --no-capture-output -n "${CONDA_ENV}" python - <<'PY' 2>&1
import importlib
import os
import sys

videodpo_repo = os.environ["VIDEODPO_REPO"]
sys.path.insert(0, videodpo_repo)
os.chdir(videodpo_repo)

required = [
    "torch",
    "pytorch_lightning",
    "omegaconf",
    "kornia",
    "open_clip",
    "transformers",
    "fairscale",
    "timm",
    "peft",
    "decord",
    "av",
]
if os.environ.get("REQUIRE_WANDB", "1") == "1":
    required.append("wandb")

missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        print(f"{name}: {type(exc).__name__}: {exc}")
        missing.append(name)

try:
    importlib.import_module("lvdm.modules.encoders.condition")
except Exception as exc:
    print(f"lvdm.modules.encoders.condition: {type(exc).__name__}: {exc}")
    missing.append("lvdm.modules.encoders.condition")

raise SystemExit(2 if missing else 0)
PY
    )"
    ENV_IMPORT_RC=$?
    if [[ "$ENV_IMPORT_RC" -eq 0 ]]; then
      ok "VideoDPO env import preflight passed in conda env: ${CONDA_ENV}"
    else
      printf '%s\n' "$ENV_IMPORT_OUTPUT" | sed '/^$/d' | sed 's/^/[DIAG] env_import_failure=/'
      fail "VideoDPO env import preflight failed in conda env ${CONDA_ENV}. Fix: CONDA_ENV=${CONDA_ENV} INSTALL_MINIMAL=1 bash DPO_finetune/scripts/videodpo_env_smoke_and_export.sh"
    fi
  else
    warn "Skipping env import preflight because conda command is not available"
  fi
else
  warn "CHECK_ENV_IMPORTS=0; skipped VideoDPO env import preflight"
fi

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
  printf 'failures:\n'
  for msg in "${FAIL_MESSAGES[@]}"; do
    printf '  - %s\n' "$msg"
  done
fi
if [[ "$WARNINGS" -gt 0 ]]; then
  printf 'warnings:\n'
  for msg in "${WARN_MESSAGES[@]}"; do
    printf '  - %s\n' "$msg"
  done
fi
if [[ "$ERRORS" -gt 0 ]]; then
  printf '[RESULT] FAIL: fix missing required assets before training/evaluation.\n'
  exit 1
fi
if [[ "$WARNINGS" -gt 0 ]]; then
  printf '[RESULT] PASS WITH WARNINGS: review warnings before long jobs.\n'
  exit 0
fi
printf '[RESULT] PASS: static health check found required assets.\n'
