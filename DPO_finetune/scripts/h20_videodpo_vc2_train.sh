#!/usr/bin/env bash
# H20 bash launcher for official VideoDPO VC2-DPO reproduction.
# It reuses the SC launcher body but supplies H20 paths and disables W&B by
# default. Use SMOKE=1 for a one-GPU 2-step smoke, then SMOKE=0 for 8 GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

export PROJECT_ROOT
export PROJECT_HOME="${PROJECT_HOME:-/home/nvme01}"
export PROJECT_DEV="${PROJECT_DEV:-/home/nvme01}"
export PROJECT_DATA="${PROJECT_DATA:-/home/nvme01}"
export VIDEODPO_REPO="${VIDEODPO_REPO:-/home/nvme01/VideoDPO}"
export LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/videodpo_vc2_dpo}"
export CONDA_ENV="${CONDA_ENV:-/home/nvme01/conda_envs/videodpo}"
export ENABLE_WANDB="${ENABLE_WANDB:-0}"
export EARLY_WANDB="${EARLY_WANDB:-0}"
export WANDB_START_EVENT="${WANDB_START_EVENT:-0}"
export PATCH_LOGGER_COMPAT="${PATCH_LOGGER_COMPAT:-1}"
export DISABLE_IMAGE_LOGGER="${DISABLE_IMAGE_LOGGER:-1}"
export PATCH_OPENCLIP_BATCH_FIRST="${PATCH_OPENCLIP_BATCH_FIRST:-1}"
export APPLY_DPO_DIAG_PATCH="${APPLY_DPO_DIAG_PATCH:-1}"
export CLEAN_DEBUG_PRINT="${CLEAN_DEBUG_PRINT:-1}"
export BETA_DPO="${BETA_DPO:-5000}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export CONFIG="${CONFIG:-configs/vc2_dpo/config.yaml}"

SMOKE="${SMOKE:-1}"
case "${SMOKE,,}" in
  1|true|yes|on)
    export NUM_GPUS="${NUM_GPUS:-1}"
    export DEVICE_LIST="${DEVICE_LIST:-0}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEVICE_LIST}}"
    export NUM_WORKERS="${NUM_WORKERS:-0}"
    export MAX_OPT_STEPS="${MAX_OPT_STEPS:-2}"
    export CKPT_EVERY="${CKPT_EVERY:-999999}"
    export RUN_NAME="${RUN_NAME:-h20-vc2-dpo-official-smoke-gpu${DEVICE_LIST//,/}_$(date +%Y%m%d_%H%M%S)}"
    ;;
  *)
    export NUM_GPUS="${NUM_GPUS:-8}"
    export DEVICE_LIST="${DEVICE_LIST:-0,1,2,3,4,5,6,7}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEVICE_LIST}}"
    export NUM_WORKERS="${NUM_WORKERS:-16}"
    export MAX_OPT_STEPS="${MAX_OPT_STEPS:-}"
    export CKPT_EVERY="${CKPT_EVERY:-499}"
    export RUN_NAME="${RUN_NAME:-h20-vc2-dpo-official-beta5000-gpu0-7_$(date +%Y%m%d_%H%M%S)}"
    ;;
esac

mkdir -p "${LOG_ROOT}"

PYTHON_BIN="${CONDA_ENV}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

RAW_VC2_DATA_YAML="${VC2_DATA_YAML:-}"
export RAW_VC2_DATA_YAML
VC2_DATA_YAML="$("${PYTHON_BIN}" - <<'PY'
import glob
import os
import sys
from pathlib import Path

import yaml

raw = os.environ.get("RAW_VC2_DATA_YAML", "").strip()
videodpo_repo = Path(os.environ["VIDEODPO_REPO"]).expanduser()
log_root = Path(os.environ["LOG_ROOT"]).expanduser()

if raw:
    candidates = [raw]
else:
    patterns = [
        str(videodpo_repo / "configs/vc2_dpo/vidpro/train_data.yaml"),
        str(Path(os.environ["PROJECT_ROOT"]) / "logs/vc2_dpo_videoinpainting_h20_gpu*/configs/train_data.yaml"),
        str(Path(os.environ["PROJECT_ROOT"]) / "logs/videodpo_vc2_dpo/*/configs/train_data.yaml"),
        "/home/nvme01/VideoDPO_runs/vc2_dpo_clean_*/configs/train_data.yaml",
        "/home/nvme01/VideoDPO_runs/vc2_dpo_diag_*/configs/train_data.yaml",
        "/home/nvme0*/VideoDPO*/configs/vc2_dpo/vidpro/train_data*.yaml",
        "/home/nvme0*/VideoDPO_runs/*/configs/train_data.yaml",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))

seen = set()
candidates = [c for c in candidates if not (c in seen or seen.add(c))]

def resolve_meta(meta, yaml_path):
    path = Path(str(meta)).expanduser()
    if path.is_absolute():
        roots = [path]
    else:
        roots = [videodpo_repo / path, yaml_path.parent / path, Path.cwd() / path]
    for root in roots:
        if (root / "metadata.json").is_file() and (root / "pair.json").is_file():
            return root.resolve()
    return None

for candidate in candidates:
    yaml_path = Path(candidate).expanduser()
    if yaml_path.is_dir() and (yaml_path / "metadata.json").is_file() and (yaml_path / "pair.json").is_file():
        log_root.mkdir(parents=True, exist_ok=True)
        out = log_root / "h20_vc2_train_data.absolute.yaml"
        out.write_text(yaml.safe_dump({"META": [str(yaml_path.resolve())]}, sort_keys=False))
        print(out)
        raise SystemExit(0)
    if not yaml_path.is_file():
        continue
    try:
        cfg = yaml.safe_load(yaml_path.read_text()) or {}
    except Exception as exc:
        print(f"[h20-vc2][warn] skip unreadable yaml {yaml_path}: {exc}", file=sys.stderr)
        continue
    roots = []
    ok = True
    for meta in cfg.get("META", []):
        root = resolve_meta(meta, yaml_path)
        if root is None:
            ok = False
            break
        roots.append(str(root))
    if ok and roots:
        log_root.mkdir(parents=True, exist_ok=True)
        out = log_root / "h20_vc2_train_data.absolute.yaml"
        out.write_text(yaml.safe_dump({"META": roots}, sort_keys=False))
        print(out)
        raise SystemExit(0)

scan_roots = [
    Path(p).expanduser()
    for p in os.environ.get("VC2_DATA_SEARCH_ROOTS", "/home/nvme01 /home/nvme02 /home/nvme03 /home/nvme04").split()
]
scan_roots = [p for p in scan_roots if p.exists()]
found_roots = []
all_pair_roots = []
archive_candidates = []
if scan_roots:
    import subprocess

    cmd = [
        "find",
        *[str(p) for p in scan_roots],
        "(",
        "-path", "*/.git", "-o",
        "-path", "*/.cache", "-o",
        "-path", "*/wandb", "-o",
        "-path", "*/miniconda3", "-o",
        "-path", "*/conda_envs", "-o",
        "-path", "*/envs",
        ")",
        "-prune",
        "-o",
        "-type", "f",
        "-name", "pair.json",
        "-print",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for line in proc.stdout.splitlines():
        root = Path(line).parent
        if not (root / "metadata.json").is_file():
            continue
        all_pair_roots.append(root.resolve())
        root_s = str(root).lower()
        if not any(token in root_s for token in ("vidpro", "vc2", "videodpo")):
            continue
        found_roots.append(root.resolve())

    archive_cmd = [
        "find",
        *[str(p) for p in scan_roots],
        "(",
        "-path", "*/.git", "-o",
        "-path", "*/.cache", "-o",
        "-path", "*/wandb", "-o",
        "-path", "*/miniconda3", "-o",
        "-path", "*/conda_envs", "-o",
        "-path", "*/envs",
        ")",
        "-prune",
        "-o",
        "-type", "f",
        "(",
        "-iname", "*vidpro*.tar", "-o",
        "-iname", "*vidpro*.tar.gz", "-o",
        "-iname", "*vidpro*.tgz", "-o",
        "-iname", "*vidpro*.zip", "-o",
        "-iname", "*vc2*.tar", "-o",
        "-iname", "*vc2*.tar.gz", "-o",
        "-iname", "*vc2*.tgz", "-o",
        "-iname", "*vc2*.zip", "-o",
        "-iname", "*videodpo*.tar", "-o",
        "-iname", "*videodpo*.tar.gz", "-o",
        "-iname", "*videodpo*.tgz", "-o",
        "-iname", "*videodpo*.zip",
        ")",
        "-print",
    ]
    archives = subprocess.run(archive_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    archive_candidates = sorted(set(archives.stdout.splitlines()))

found_roots = sorted({str(p): p for p in found_roots}.values(), key=lambda p: str(p))
if found_roots:
    log_root.mkdir(parents=True, exist_ok=True)
    out = log_root / "h20_vc2_train_data.absolute.yaml"
    out.write_text(yaml.safe_dump({"META": [str(p) for p in found_roots]}, sort_keys=False))
    print(f"[h20-vc2][data] generated absolute yaml from filesystem scan: {out}", file=sys.stderr)
    for root in found_roots:
        print(f"[h20-vc2][data] META={root}", file=sys.stderr)
    print(out)
    raise SystemExit(0)

all_pair_roots = sorted({str(p): p for p in all_pair_roots}.values(), key=lambda p: str(p))
if all_pair_roots:
    print("[h20-vc2][diagnostic] found pair.json+metadata.json roots, but their paths do not look like VC2/vidpro:", file=sys.stderr)
    for root in all_pair_roots[:50]:
        print(f"  {root}", file=sys.stderr)
    print("[h20-vc2][diagnostic] If one is the intended VideoDPO VC2 root, rerun with VC2_DATA_YAML=/that/root", file=sys.stderr)

print("[h20-vc2][error] no valid VC2 train_data yaml or dataset root found.", file=sys.stderr)
print("[h20-vc2][error] Set VC2_DATA_YAML to a yaml whose META roots contain metadata.json and pair.json,", file=sys.stderr)
print("[h20-vc2][error] or set VC2_DATA_SEARCH_ROOTS to directories that contain the extracted vidpro/vc2 dataset.", file=sys.stderr)
print("[h20-vc2][error] checked candidates:", file=sys.stderr)
for candidate in candidates:
    print(f"  {candidate}", file=sys.stderr)
print("[h20-vc2][error] scanned roots:", file=sys.stderr)
for root in scan_roots:
    print(f"  {root}", file=sys.stderr)
if archive_candidates:
    print("[h20-vc2][diagnostic] possible compressed dataset archives:", file=sys.stderr)
    for path in archive_candidates[:50]:
        print(f"  {path}", file=sys.stderr)
raise SystemExit(2)
PY
)"
export VC2_DATA_YAML

echo "[h20-vc2] project_root=${PROJECT_ROOT}"
echo "[h20-vc2] videodpo_repo=${VIDEODPO_REPO}"
echo "[h20-vc2] vc2_data_yaml=${VC2_DATA_YAML}"
echo "[h20-vc2] log_root=${LOG_ROOT}"
echo "[h20-vc2] conda_env=${CONDA_ENV}"
echo "[h20-vc2] smoke=${SMOKE} num_gpus=${NUM_GPUS} device_list=${DEVICE_LIST} max_opt_steps=${MAX_OPT_STEPS:-official-config-max_epochs}"

exec bash "${PROJECT_ROOT}/DPO_finetune/scripts/sc_videodpo_vc2_train.sbatch"
