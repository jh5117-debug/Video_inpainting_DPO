#!/usr/bin/env bash
set -u
set -o pipefail

REPO_ROOT="/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp49_rose_adapter"
BASE="/mnt/nas/hj/H20_Video_inpainting_DPO"
ROSE_CODE="$BASE/third_party/ROSE/Kunbyte-AI_ROSE"
SPACE_CODE="$BASE/third_party/ROSE_HF_Space/Kunbyte_ROSE"
ROSE_MODEL="$BASE/weights/rose/Kunbyte_ROSE"
WAN_MODEL="$BASE/weights/rose/Wan2.1-Fun-1.3B-InP"
LOGROOT="$BASE/logs/autoresearch/exp49_pai_rose_adapter_feasibility"
RUNTIME="$BASE/runtime/exp49_pai_rose_adapter_feasibility"
ENV_DIR="/home/hj/venvs/rose_exp49_py310"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGROOT/milestone_c_env_smoke_${STAMP}.log"
MONITOR="$LOGROOT/monitor_5min.csv"
mkdir -p "$LOGROOT" "$RUNTIME" "$REPO_ROOT/reports" "$(dirname "$ENV_DIR")"
exec > >(tee -a "$LOG") 2>&1

cd "$REPO_ROOT"

echo "# Exp49 Milestone C ROSE environment smoke"
echo "time=$(date -Ins)"
echo "hostname=$(hostname)"
echo "branch=$(git branch --show-current)"
echo "commit=$(git rev-parse HEAD)"
echo "env_dir=$ENV_DIR"
echo

if [ ! -f "$MONITOR" ]; then
  echo "time,hostname,branch,commit,milestone,GPU,PID,PGID,VRAM,util,sample_progress,step,loss,grad_norm,NaN_Inf,OOM_CUDA_Xid,last_error,checkpoint,next_action" > "$MONITOR"
fi
printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
  "$(date -Ins)" "$(hostname)" "$(git branch --show-current)" "$(git rev-parse HEAD)" \
  "C_env_start" "0" "$$" "$(ps -o pgid= $$ | tr -d ' ')" "NA" "NA" "env" "0" "NA" "NA" "none" "none" "none" "none" "create env/import smoke" >> "$MONITOR"

echo "## git/readback"
git fetch --all --prune || true
git branch --show-current
git rev-parse HEAD
git status --short
git log -8 --oneline
git diff --stat
git diff --check

echo "## gpu preflight"
hostname
date -Ins
nvidia-smi || true
nvidia-smi pmon -i 0,1 -c 1 || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv || true
ps -eo user,pid,ppid,pgid,sid,etime,stat,cmd --sort=pid | tail -120 || true

echo "## python availability"
python3 --version || true
python3.12 --version || true
python3.12 -m pip --version || true

if [ ! -d "$ENV_DIR" ]; then
  python3 -m venv --system-site-packages "$ENV_DIR"
fi
. "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "## pip install ROSE core requirements"
set +e
python -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r "$ROSE_CODE/requirements.txt" \
  "diffusers==0.31.0" \
  "transformers==4.46.2"
pip_rc=$?
set -e
echo "pip_install_rc=$pip_rc"

REPORT="$REPO_ROOT/reports/exp49_rose_env_smoke.md"
CSV="$REPO_ROOT/reports/exp49_rose_env_smoke.csv"
SUMMARY="$REPO_ROOT/reports/exp49_rose_env_smoke_summary.json"
IMPORT_LOG="$RUNTIME/exp49_rose_env_import_smoke.log"

echo "## import and CUDA smoke"
set +e
ROSE_CODE="$ROSE_CODE" SPACE_CODE="$SPACE_CODE" ROSE_MODEL="$ROSE_MODEL" WAN_MODEL="$WAN_MODEL" CUDA_VISIBLE_DEVICES=0 python - <<'PY' > "$IMPORT_LOG" 2>&1
import importlib, json, os, sys
from pathlib import Path

rose_code = Path(os.environ["ROSE_CODE"])
space_code = Path(os.environ["SPACE_CODE"])
rose_model = Path(os.environ["ROSE_MODEL"])
wan_model = Path(os.environ["WAN_MODEL"])
sys.path.insert(0, str(rose_code))

rows = []
def add(name, status, note=""):
    rows.append({"check": name, "status": status, "note": note})
    print(f"{status}\t{name}\t{note}", flush=True)

add("python_version", "PASS", sys.version.replace("\n", " "))

mods = [
    "PIL", "einops", "safetensors", "timm", "tomesd", "torchdiffeq",
    "torchsde", "decord", "datasets", "numpy", "skimage", "cv2",
    "omegaconf", "sentencepiece", "albumentations", "imageio", "bs4",
    "ftfy", "func_timeout", "accelerate", "diffusers", "transformers",
]
for mod in mods:
    try:
        m = importlib.import_module(mod)
        add(f"import:{mod}", "PASS", getattr(m, "__version__", ""))
    except Exception as e:
        add(f"import:{mod}", "FAIL", repr(e))

try:
    import torch
    add("torch", "PASS", f"{torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
except Exception as e:
    add("torch", "FAIL", repr(e))
    torch = None

for mod in [
    "rose.models",
    "rose.models.wan_transformer3d",
    "rose.models.wan_vae",
    "rose.models.wan_text_encoder",
    "rose.models.wan_image_encoder",
    "rose.models.diff_mask_predictor",
    "rose.pipeline.pipeline_wan_fun_inpaint",
    "rose.utils.utils",
    "rose.data.dataset_image_video",
    "inference",
]:
    try:
        importlib.import_module(mod)
        add(f"rose_import:{mod}", "PASS", "")
    except Exception as e:
        add(f"rose_import:{mod}", "FAIL", repr(e))

required_paths = [
    rose_model / "config.json",
    rose_model / "diffusion_pytorch_model.safetensors",
    wan_model / "config.json",
    wan_model / "configuration.json",
    wan_model / "diffusion_pytorch_model.safetensors",
    wan_model / "Wan2.1_VAE.pth",
    wan_model / "models_t5_umt5-xxl-enc-bf16.pth",
    wan_model / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    wan_model / "google/umt5-xxl/tokenizer.json",
    wan_model / "xlm-roberta-large/tokenizer.json",
]
for p in required_paths:
    add(f"path:{p}", "PASS" if p.exists() else "FAIL", str(p.stat().st_size) if p.exists() else "missing")

try:
    import torch
    if torch.cuda.is_available():
        x = torch.randn((256, 256), device="cuda", dtype=torch.float16)
        y = x @ x
        torch.cuda.synchronize()
        add("cuda_matmul_gpu0", "PASS", f"mean={float(y.float().mean().cpu()):.6f}")
    else:
        add("cuda_matmul_gpu0", "FAIL", "cuda unavailable")
except Exception as e:
    add("cuda_matmul_gpu0", "FAIL", repr(e))

summary = {
    "rows": rows,
    "fail_count": sum(1 for r in rows if r["status"] != "PASS"),
}
Path(os.environ.get("SUMMARY_PATH", "/tmp/exp49_rose_env_import_summary.json")).write_text(json.dumps(summary, indent=2))
PY
smoke_rc=$?
set -e
echo "smoke_rc=$smoke_rc"

SUMMARY_PATH_TMP="/tmp/exp49_rose_env_import_summary.json"
python - <<PY
import csv, json, os, pathlib, subprocess
import_log = pathlib.Path("$IMPORT_LOG")
rows = []
for line in import_log.read_text(errors="replace").splitlines():
    parts = line.split("\t", 2)
    if len(parts) == 3 and parts[0] in {"PASS", "FAIL"}:
        rows.append({"status": parts[0], "check": parts[1], "note": parts[2]})
with open("$CSV", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["check", "status", "note"])
    w.writeheader()
    for r in rows:
        w.writerow({"check": r["check"], "status": r["status"], "note": r["note"]})
fail = [r for r in rows if r["status"] != "PASS"]
python_version = subprocess.getoutput("$ENV_DIR/bin/python --version")
torch_line = next((r["note"] for r in rows if r["check"] == "torch"), "")
official_python = subprocess.getoutput("python3.12 --version 2>/dev/null || true")
if fail:
    status = "ROSE_ENV_BLOCKED"
elif not python_version.startswith("Python 3.12"):
    status = "ROSE_ENV_PARTIAL"
else:
    status = "ROSE_ENV_READY"
summary = {
    "generated": subprocess.getoutput("date -Ins"),
    "hostname": subprocess.getoutput("hostname"),
    "status": status,
    "env_path": "$ENV_DIR",
    "python_version": python_version,
    "official_python_available": official_python,
    "pip_install_rc": $pip_rc,
    "smoke_rc": $smoke_rc,
    "fail_count": len(fail),
    "failed_checks": fail,
    "torch": torch_line,
    "notes": "Python 3.12 exists but has no pip/torch; isolated Python 3.10 venv uses system Torch 2.6.0+cu126.",
}
pathlib.Path("$SUMMARY").write_text(json.dumps(summary, indent=2))
PY

final_status="$(python - <<PY
import json
print(json.load(open("$SUMMARY"))["status"])
PY
)"

cat > "$REPORT" <<MD
# Exp49 ROSE Environment Smoke

Status: \`$final_status\`

Generated: $(date -Ins)
Host: $(hostname)
Branch: $(git branch --show-current)
Commit: $(git rev-parse HEAD)

## Environment

- Env path: \`$ENV_DIR\`
- Python used: \`$("$ENV_DIR/bin/python" --version 2>&1)\`
- Official requested Python: ROSE README says Python 3.12.
- PAI Python 3.12: \`$(python3.12 --version 2>&1 || true)\`
- Python 3.12 pip/torch: unavailable in this image.
- Torch/CUDA: \`$("$ENV_DIR/bin/python" - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
)\`

## Install

ROSE core requirements were installed into the isolated venv. Base/global Python was not modified.

- pip install rc: \`$pip_rc\`
- smoke rc: \`$smoke_rc\`

## Smoke Checks

- CSV: \`reports/exp49_rose_env_smoke.csv\`
- Summary JSON: \`reports/exp49_rose_env_smoke_summary.json\`
- Import log: \`$IMPORT_LOG\`

## GPU

A tiny CUDA matmul smoke was run on GPU0 only. No model inference, training, optimizer step, checkpoint update, or H20 action was performed.
MD

cat > experiment_registry/exp49_pai_rose_adapter_feasibility/status.md <<MD
# Exp49 Status

Current status: \`$final_status\`

Milestone C created an isolated Python venv and ran ROSE import/CUDA smoke. Python 3.12 is present on PAI but lacks pip/torch, so the smoke used Python 3.10 with system Torch 2.6.0+cu126.

No inference, training, optimizer step, DPO, or H20 action was run.
MD

python - <<PY
from pathlib import Path
p = Path("experiment_registry/exp49_pai_rose_adapter_feasibility/results.tsv")
lines = p.read_text().splitlines()
out = []
for line in lines:
    if line.startswith("C_env\t"):
        out.append("C_env\t$final_status\tIsolated Python 3.10 venv import/CUDA smoke; Python 3.12 unavailable for torch/pip.")
    else:
        out.append(line)
p.write_text("\n".join(out) + "\n")
PY

cat > experiment_registry/exp49_pai_rose_adapter_feasibility/metric_summary.md <<MD
# Exp49 Metric Summary

Milestone C computed no inpainting metrics. Environment/import/CUDA smoke status: \`$final_status\`.
MD

cat > experiment_registry/exp49_pai_rose_adapter_feasibility/qualitative_summary.md <<MD
# Exp49 Qualitative Summary

No ROSE visual evidence has been generated or reviewed yet.

Milestone C only ran dependency import and CUDA smoke.
MD

EXP49_STATUS="$final_status" EXP49_ENV_DIR="$ENV_DIR" python - <<'PY'
from pathlib import Path
import os
status = os.environ["EXP49_STATUS"]
env_dir = os.environ["EXP49_ENV_DIR"]
updates = {
 "PRD/00_current_status.md": f"""## 2026-06-30 Exp49 ROSE Environment Smoke

Exp49 Milestone C created an isolated Python environment and ran ROSE import/CUDA smoke. Status: `{status}`. Python 3.12 exists on PAI but has no pip/torch, so the smoke used Python 3.10 with system Torch 2.6.0+cu126. No inference, training, optimizer step, DPO, or H20 action was run.
""",
 "PRD/01_experiment_matrix.md": f"""## 2026-06-30 Exp49 Environment Update

| Experiment | Milestone | Status | Notes |
| --- | --- | --- | --- |
| `exp49_pai_rose_adapter_feasibility` | C env | `{status}` | Isolated venv import/CUDA smoke; no inference/training; Python 3.12 lacks pip/torch on PAI. |
""",
 "PRD/46_exp49_pai_rose_adapter_feasibility.md": f"""## Milestone C Update - 2026-06-30

Status: `{status}`.

An isolated venv at `{env_dir}` was created and used for ROSE dependency/import/CUDA smoke. Python 3.12 is present on PAI but lacks pip/torch, so the smoke used Python 3.10 with system Torch 2.6.0+cu126. Reports are available at `reports/exp49_rose_env_smoke.md`, `reports/exp49_rose_env_smoke.csv`, and `reports/exp49_rose_env_smoke_summary.json`.

No inference, training, optimizer step, DPO, or H20 action was run.
""",
}
for name, text in updates.items():
    p = Path(name)
    cur = p.read_text()
    marker = text.strip().split("\n", 1)[0]
    if marker not in cur:
        p.write_text(cur.rstrip() + "\n\n" + text.strip() + "\n")
PY

printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
  "$(date -Ins)" "$(hostname)" "$(git branch --show-current)" "$(git rev-parse HEAD)" \
  "C_env_done" "0" "$$" "$(ps -o pgid= $$ | tr -d ' ')" "tiny" "tiny" "env" "0" "NA" "NA" "none" "none" "$final_status" "none" "commit milestone C" >> "$MONITOR"

echo
cat "$REPORT"
echo
echo "## git status after Milestone C"
git status --short
