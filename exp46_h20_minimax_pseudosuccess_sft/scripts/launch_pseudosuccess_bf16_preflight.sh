#!/usr/bin/env bash
set -euo pipefail

WORKTREE="${WORKTREE:-/home/nvme01/H20_Video_inpainting_DPO_exp46_minimax_pseudosuccess_sft}"
PROJECT_ROOT="${PROJECT_ROOT:-${WORKTREE}}"
MINIMAX_REPO="${MINIMAX_REPO:-/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax/pai_abs/mnt/nas/hj/official_repos/MiniMax-Remover_28e12b4}"
MODEL_DIR="${MODEL_DIR:-/home/nvme01/H20_Video_inpainting_DPO/weights/minimax_remover/current}"
MANIFEST="${MANIFEST:-${WORKTREE}/manifests/exp46_runner_pseudosuccess_train.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp46_h20_minimax_pseudosuccess_sft/bf16_preflight_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-/home/nvme01/H20_Video_inpainting_DPO/logs/autoresearch/exp46_h20_minimax_pseudosuccess_sft}"
REPORTS_DIR="${REPORTS_DIR:-${WORKTREE}/reports}"
CONDA_BIN="${CONDA_BIN:-/home/nvme01/miniconda3/bin/conda}"
WAN_ENV="${WAN_ENV:-/home/nvme01/miniconda3/envs/wan}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_BIN} run --no-capture-output -p ${WAN_ENV} python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_BIN} run --no-capture-output -p ${WAN_ENV} torchrun}"
SCRIPT="${WORKTREE}/exp46_h20_minimax_pseudosuccess_sft/scripts/run_pseudosuccess_bf16_preflight.py"
SEED="${SEED:-20260629}"

export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export XFORMERS_DISABLED=1
export DISABLE_XFORMERS=1
export FLASH_ATTENTION_FORCE_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUTPUT_ROOT}/logs" "${LOG_ROOT}" "${REPORTS_DIR}"
BRANCH="$(git -C "${WORKTREE}" branch --show-current)"
COMMIT="$(git -C "${WORKTREE}" rev-parse HEAD)"
MANIFEST_SHA="$(sha256sum "${MANIFEST}" | cut -d ' ' -f1)"
CONFIG_SHA="$(sha256sum "${SCRIPT}" | cut -d ' ' -f1)"
{
  echo "time,hostname,branch,commit,gpu,pid,pgid,milestone,case_name,manifest_sha,config_sha,output_root,vram,util,last_error,next_action"
  echo "$(date -Ins),$(hostname),${BRANCH},${COMMIT},all,$$,$(ps -o pgid= $$ | xargs),D_preflight,START,${MANIFEST_SHA},${CONFIG_SHA},${OUTPUT_ROOT},NA,NA,,run_P0_P7"
} > "${OUTPUT_ROOT}/heartbeat.csv"

hostname | tee "${OUTPUT_ROOT}/hostname.txt"
date -Ins | tee "${OUTPUT_ROOT}/started_at.txt"
printf "%s\n" "branch=${BRANCH}" "commit=${COMMIT}" "manifest=${MANIFEST}" "manifest_sha=${MANIFEST_SHA}" "config_sha=${CONFIG_SHA}" "pid=$$" "pgid=$(ps -o pgid= $$ | xargs)" "output_root=${OUTPUT_ROOT}" | tee "${OUTPUT_ROOT}/runtime_snapshot.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee "${OUTPUT_ROOT}/gpu_before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "${OUTPUT_ROOT}/compute_before.csv" || true

common_args=(
  --repo-dir "${MINIMAX_REPO}"
  --project-root "${PROJECT_ROOT}"
  --model-dir "${MODEL_DIR}"
  --manifest "${MANIFEST}"
  --output-root "${OUTPUT_ROOT}"
  --seed "${SEED}"
)

for case_name in P0 P1 P2 P3 P4 P5; do
  echo "[exp46-preflight] ${case_name} GPU0"
  CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} "${SCRIPT}" --case "${case_name}" "${common_args[@]}" 2>&1 | tee "${OUTPUT_ROOT}/logs/${case_name}.log"
done

echo "[exp46-preflight] P6 DDP2 GPU0,1"
CUDA_VISIBLE_DEVICES=0,1 ${TORCHRUN_BIN} --standalone --nnodes=1 --nproc_per_node=2 "${SCRIPT}" --case P6 "${common_args[@]}" 2>&1 | tee "${OUTPUT_ROOT}/logs/P6.log"

echo "[exp46-preflight] P7 DDP8 GPU0-7"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ${TORCHRUN_BIN} --standalone --nnodes=1 --nproc_per_node=8 "${SCRIPT}" --case P7 "${common_args[@]}" 2>&1 | tee "${OUTPUT_ROOT}/logs/P7.log"

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee "${OUTPUT_ROOT}/gpu_after.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "${OUTPUT_ROOT}/compute_after.csv" || true

echo "${OUTPUT_ROOT}" | tee "${OUTPUT_ROOT}/output_root.txt"
${PYTHON_BIN} - <<'PY'
import csv, json, math
from pathlib import Path
out=Path("${OUTPUT_ROOT}")
reports=Path("${REPORTS_DIR}")
rows=[]
for case_dir in sorted(out.glob("P*")):
    for result_path in sorted(case_dir.glob("rank*.json")):
        obj=json.loads(result_path.read_text())
        rows.append(obj)
status="EXP46_BF16_SAFE_READY" if rows and all(r.get("status")=="PASS" for r in rows) and any(r.get("case")=="P7" for r in rows) else "EXP46_ENV_BLOCKED"
keys=[]
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
with (reports/"exp46_bf16_pseudosuccess_preflight.csv").open("w", newline="") as f:
    w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
summary={"status":status,"output_root":str(out),"rows":len(rows),"cases":sorted(set(r.get("case") for r in rows)),"failed":[r for r in rows if r.get("status")!="PASS"],"training_run":False,"optimizer_step":False,"manifest":"${MANIFEST}","manifest_sha":"${MANIFEST_SHA}","config_sha":"${CONFIG_SHA}"}
(reports/"exp46_bf16_pseudosuccess_preflight_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
lines=["# Exp46 Pseudo-Success BF16 Preflight", "", f"Status: {status}", "", f"Output root: `{out}`", "", "| case | ranks | world_size | status | loss | grad_norm | checkpoint | peak MiB |", "| --- | ---: | ---: | --- | ---: | ---: | --- | ---: |"]
for r in rows:
    lines.append(f"| {r.get(case)} | {r.get(rank)} | {r.get(world_size)} | {r.get(status)} | {r.get(loss,)} | {r.get(grad_norm,)} | {r.get(checkpoint_status,)} | {r.get(max_memory_allocated,)} |")
lines += ["", "Policy: bf16 DiT/LoRA, fp32 VAE and loss/reduction, no optimizer step in P0-P7, P5 checkpoint save/reload dry-run only.", "", "No training run, no optimizer step, no PAI write/GPU, no quality claim."]
(reports/"exp46_bf16_pseudosuccess_preflight.md").write_text("\n".join(lines)+"\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
