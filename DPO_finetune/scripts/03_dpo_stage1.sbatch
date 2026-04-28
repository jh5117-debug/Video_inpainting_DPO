#!/bin/bash
#SBATCH --job-name=DPO_S1
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/dpo-stage1-%j.out

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${PROJECT_DEV:-}" ]]; then
    export PROJECT_ROOT="${PROJECT_DEV}/H20_Video_inpainting_DPO"
  elif [[ -n "${PROJECT_HOME:-}" ]]; then
    export PROJECT_ROOT="${PROJECT_HOME}/dev/H20_Video_inpainting_DPO"
  else
    export PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
exec bash "${PROJECT_ROOT}/training/dpo/scripts/03_dpo_stage1.sbatch" "$@"
