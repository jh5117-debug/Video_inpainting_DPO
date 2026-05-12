#!/usr/bin/env bash
# Build/reuse a VideoDPO-compatible conda env, run a CPU-only import/config
# smoke, and export environment manifests for SC reuse.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VIDEODPO_REPO="${VIDEODPO_REPO:-${PROJECT_ROOT}/external/VideoDPO}"
CONDA_ENV="${CONDA_ENV:-${VIDEODPO_CONDA_ENV:-videodpo}}"
FALLBACK_CONDA_ENV="${FALLBACK_CONDA_ENV:-}"
CREATE_ENV="${CREATE_ENV:-0}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"
INSTALL_MINIMAL="${INSTALL_MINIMAL:-0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
EXPORT_ENV="${EXPORT_ENV:-1}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/env_exports}"
ENV_BASENAME="${ENV_BASENAME:-videodpo_${CONDA_ENV}_$(hostname)_$(date -u +%Y%m%d_%H%M%S)}"

if [[ -n "${CONDA_BASE:-}" && -x "${CONDA_BASE}/bin/conda" ]]; then
  :
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$("${CONDA_EXE}" info --base)"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  echo "[videodpo-env][error] conda not found; set CONDA_EXE or CONDA_BASE." >&2
  exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"

env_exists() {
  conda env list 2>/dev/null | awk '{print $1}' | grep -Fxq "$1"
}

if ! env_exists "${CONDA_ENV}"; then
  if [[ "${CREATE_ENV}" == "1" ]]; then
    echo "[videodpo-env] creating conda env: ${CONDA_ENV} python=${PYTHON_VERSION}"
    conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
  elif [[ -n "${FALLBACK_CONDA_ENV}" ]] && env_exists "${FALLBACK_CONDA_ENV}"; then
    echo "[videodpo-env][warn] env ${CONDA_ENV} not found; using fallback ${FALLBACK_CONDA_ENV}"
    CONDA_ENV="${FALLBACK_CONDA_ENV}"
  else
    echo "[videodpo-env][error] conda env not found: ${CONDA_ENV}" >&2
    echo "[videodpo-env][error] Either set CONDA_ENV to an existing env, or run CREATE_ENV=1 INSTALL_REQUIREMENTS=1." >&2
    exit 1
  fi
fi

if [[ ! -f "${VIDEODPO_REPO}/requirements.txt" ]]; then
  echo "[videodpo-env][error] VideoDPO requirements missing: ${VIDEODPO_REPO}/requirements.txt" >&2
  echo "[videodpo-env][error] Run: git submodule update --init --recursive" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
NORMALIZED_REQ="${OUT_DIR}/videodpo_requirements.normalized.txt"
sed -e 's/^tenosrboard$/tensorboard/' -e 's/[[:space:]]*$//' "${VIDEODPO_REPO}/requirements.txt" > "${NORMALIZED_REQ}"
MINIMAL_REQ="${OUT_DIR}/videodpo_requirements.minimal_no_torch.txt"
cat > "${MINIMAL_REQ}" <<'REQ'
setuptools<81
pytorch-lightning==1.9.5
omegaconf
einops
tqdm
PyYAML
decord==0.6.0
av
open_clip_torch
transformers
fairscale
timm
peft==0.13.2
kornia
wandb
REQ

if [[ "${INSTALL_REQUIREMENTS}" == "1" ]]; then
  echo "[videodpo-env] installing VideoDPO requirements into ${CONDA_ENV}"
  conda run --no-capture-output -n "${CONDA_ENV}" python -m pip install -U pip wheel setuptools
  conda run --no-capture-output -n "${CONDA_ENV}" python -m pip install -r "${NORMALIZED_REQ}"
fi

if [[ "${INSTALL_MINIMAL}" == "1" ]]; then
  echo "[videodpo-env] installing minimal VideoDPO smoke requirements into ${CONDA_ENV}"
  echo "[videodpo-env] minimal list intentionally excludes torch/torchvision/numpy/xformers"
  echo "[videodpo-env] setuptools is included because pytorch_lightning 1.9 imports pkg_resources"
  conda run --no-capture-output -n "${CONDA_ENV}" python -m pip install -U pip wheel "setuptools<81"
  conda run --no-capture-output -n "${CONDA_ENV}" python -m pip install -r "${MINIMAL_REQ}"
fi

echo "[videodpo-env] smoke env=${CONDA_ENV}"
PROJECT_ROOT="${PROJECT_ROOT}" VIDEODPO_REPO="${VIDEODPO_REPO}" \
conda run --no-capture-output -n "${CONDA_ENV}" python - <<'PY'
import importlib
import os
import sys

project_root = os.environ["PROJECT_ROOT"]
videodpo_repo = os.environ["VIDEODPO_REPO"]
sys.path.insert(0, videodpo_repo)
os.chdir(videodpo_repo)

print(f"[smoke] python={sys.version.split()[0]}")
required = [
    "torch",
    "pytorch_lightning",
    "omegaconf",
    "einops",
    "tqdm",
    "numpy",
    "yaml",
    "decord",
    "av",
    "open_clip",
    "transformers",
    "fairscale",
    "timm",
    "peft",
    "kornia",
    "wandb",
]
missing = []
for name in required:
    try:
        module = importlib.import_module(name)
        print(f"[smoke][OK] {name} {getattr(module, '__version__', '')}".rstrip())
    except Exception as exc:
        print(f"[smoke][FAIL] {name}: {type(exc).__name__}: {exc}")
        missing.append(name)

if missing:
    print("[smoke][ERROR] missing/import-failing modules:", ", ".join(missing))
    print("[smoke][HINT] rerun: CONDA_ENV=<env> INSTALL_MINIMAL=1 bash DPO_finetune/scripts/videodpo_env_smoke_and_export.sh")
    print("[smoke][HINT] if pytorch_lightning fails on pkg_resources, install setuptools<81 in that env")
    raise SystemExit(2)

try:
    importlib.import_module("xformers")
    print("[smoke][OK] xformers")
except Exception as exc:
    print(f"[smoke][WARN] xformers optional import failed: {type(exc).__name__}: {exc}")

from omegaconf import OmegaConf
from utils.common_utils import instantiate_from_config  # noqa: F401
import lvdm.modules.attention  # noqa: F401
import lvdm.modules.encoders.condition  # noqa: F401
import lvdm.models.ddpm3d  # noqa: F401

cfg = OmegaConf.load("configs/vc2_dpo/config.yaml")
assert cfg.model.target == "lvdm.models.ddpm3d.LatentDiffusion", cfg.model.target
assert os.path.isfile("utils/create_ref_model.py")
assert os.path.isfile("prompts/vbench_standard_prompts.txt")
print("[smoke][OK] VideoDPO config/code smoke passed")
PY

if [[ "${EXPORT_ENV}" == "1" ]]; then
  ENV_YAML="${OUT_DIR}/${ENV_BASENAME}.environment.yml"
  PIP_FREEZE="${OUT_DIR}/${ENV_BASENAME}.pip_freeze.txt"
  echo "[videodpo-env] exporting ${ENV_YAML}"
  conda env export -n "${CONDA_ENV}" --no-builds | sed '/^prefix:/d' > "${ENV_YAML}"
  echo "[videodpo-env] exporting ${PIP_FREEZE}"
  conda run --no-capture-output -n "${CONDA_ENV}" python -m pip freeze > "${PIP_FREEZE}"
  echo "[videodpo-env] wrote:"
  ls -lh "${ENV_YAML}" "${PIP_FREEZE}"
fi

echo "[videodpo-env] done"
