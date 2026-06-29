#!/usr/bin/env bash
set -u
set -o pipefail

REPO_ROOT="/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp49_rose_adapter"
BASE="/mnt/nas/hj/H20_Video_inpainting_DPO"
THIRD="$BASE/third_party/ROSE"
SPACE_ROOT="$BASE/third_party/ROSE_HF_Space"
WEIGHTS="$BASE/weights/rose"
DATASET="$BASE/data/external/rose_dataset"
OUT="$BASE/experiments/dpo/exp49_pai_rose_adapter_feasibility"
LOGROOT="$BASE/logs/autoresearch/exp49_pai_rose_adapter_feasibility"
RUNTIME="$BASE/runtime/exp49_pai_rose_adapter_feasibility"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGROOT/milestone_b_assets_mirror_${STAMP}.log"
MONITOR="$LOGROOT/monitor_5min.csv"
HF_MIRROR="https://hf-mirror.com"
mkdir -p "$THIRD" "$SPACE_ROOT" "$WEIGHTS" "$DATASET" "$OUT" "$LOGROOT" "$RUNTIME" "$REPO_ROOT/reports"
exec > >(tee -a "$LOG") 2>&1

cd "$REPO_ROOT"

echo "# Exp49 Milestone B ROSE asset download (HF mirror)"
echo "time=$(date -Ins)"
echo "hostname=$(hostname)"
echo "branch=$(git branch --show-current 2>/dev/null || true)"
echo "commit=$(git rev-parse HEAD 2>/dev/null || true)"
echo "uid=$(id)"
echo "hf_mirror=$HF_MIRROR"
echo

if [ ! -f "$MONITOR" ]; then
  echo "time,hostname,branch,commit,milestone,GPU,PID,PGID,VRAM,util,sample_progress,step,loss,grad_norm,NaN_Inf,OOM_CUDA_Xid,last_error,checkpoint,next_action" > "$MONITOR"
fi
printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
  "$(date -Ins)" "$(hostname)" "$(git branch --show-current 2>/dev/null || true)" "$(git rev-parse HEAD 2>/dev/null || true)" \
  "B_assets_mirror_start" "none" "$$" "$(ps -o pgid= $$ | tr -d ' ')" "NA" "NA" "0" "0" "NA" "NA" "none" "none" "none" "none" "hf mirror download assets" >> "$MONITOR"

run_step() {
  local name="$1"
  shift
  echo
  echo "## STEP $name START $(date -Ins)"
  set +e
  "$@"
  local rc=$?
  set -e
  echo "## STEP $name END rc=$rc $(date -Ins)"
  return $rc
}

record_status() {
  local asset="$1" status="$2" path="$3" note="$4"
  printf '%s\t%s\t%s\t%s\n' "$asset" "$status" "$path" "$note" >> "$RUNTIME/exp49_asset_status.tsv"
}

printf 'asset\tstatus\tpath\tnote\n' > "$RUNTIME/exp49_asset_status.tsv"

echo "## disk"
df -h /mnt/nas /mnt/workspace "$BASE" || true
for p in "$THIRD" "$SPACE_ROOT" "$WEIGHTS" "$DATASET" "$OUT" "$LOGROOT" "$RUNTIME"; do
  ls -ld "$p" || true
  test -w "$p" && echo "WRITABLE $p" || echo "NOT_WRITABLE $p"
done

echo "## tool versions"
git --version || true
hf --version || true
python3 --version || true
python3 - <<'PY' || true
try:
    import huggingface_hub
    print("huggingface_hub", huggingface_hub.__version__)
except Exception as e:
    print("huggingface_hub_unavailable", repr(e))
PY

CODE_DIR="$THIRD/Kunbyte-AI_ROSE"
if [ -d "$CODE_DIR/.git" ]; then
  run_step code_fetch timeout 60 git -C "$CODE_DIR" fetch --all --prune || true
  run_step code_checkout git -C "$CODE_DIR" checkout main || true
  run_step code_pull timeout 60 git -C "$CODE_DIR" pull --ff-only || true
else
  run_step code_clone git clone https://github.com/Kunbyte-AI/ROSE.git "$CODE_DIR" || true
fi
if [ -d "$CODE_DIR/.git" ]; then
  code_sha="$(git -C "$CODE_DIR" rev-parse HEAD 2>/dev/null || true)"
  record_status code READY "$CODE_DIR" "$code_sha"
else
  record_status code BLOCKED "$CODE_DIR" "clone failed"
fi

SPACE_DIR="$SPACE_ROOT/Kunbyte_ROSE"
if [ -d "$SPACE_DIR/.git" ]; then
  run_step hf_space_fetch git -C "$SPACE_DIR" fetch --all --prune || true
  run_step hf_space_pull git -C "$SPACE_DIR" pull --ff-only || true
else
  rm -rf "$SPACE_DIR"
  run_step hf_space_git_clone timeout 1800 git clone --depth 1 "$HF_MIRROR/spaces/Kunbyte/ROSE" "$SPACE_DIR" || true
fi
if [ -d "$SPACE_DIR" ] && [ -n "$(find "$SPACE_DIR" -type f 2>/dev/null | head -1)" ]; then
  space_sha="$(git -C "$SPACE_DIR" rev-parse HEAD 2>/dev/null || true)"
  record_status hf_space READY "$SPACE_DIR" "$space_sha"
else
  record_status hf_space BLOCKED "$SPACE_DIR" "mirror git clone failed or empty"
fi

MODEL_DIR="$WEIGHTS/Kunbyte_ROSE"
mkdir -p "$MODEL_DIR"
run_step hf_model_download timeout 7200 bash -lc "HF_ENDPOINT=$HF_MIRROR hf download Kunbyte/ROSE --repo-type model --local-dir '$MODEL_DIR' --max-workers 4" || true
if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/diffusion_pytorch_model.safetensors" ]; then
  record_status hf_model READY "$MODEL_DIR" "hf mirror model download"
elif [ -d "$MODEL_DIR" ] && [ -n "$(find "$MODEL_DIR" -type f 2>/dev/null | head -1)" ]; then
  record_status hf_model PARTIAL "$MODEL_DIR" "hf mirror partial model download"
else
  record_status hf_model BLOCKED "$MODEL_DIR" "hf mirror model failed or empty"
fi

WAN_DIR="$WEIGHTS/Wan2.1-Fun-1.3B-InP"
mkdir -p "$WAN_DIR"
run_step wan_base_download timeout 10800 bash -lc "HF_ENDPOINT=$HF_MIRROR hf download alibaba-pai/Wan2.1-Fun-1.3B-InP --repo-type model --local-dir '$WAN_DIR' --max-workers 4" || true
if [ -d "$WAN_DIR" ] && [ -f "$WAN_DIR/diffusion_pytorch_model.safetensors" ] && [ -f "$WAN_DIR/Wan2.1_VAE.pth" ]; then
  record_status wan_base READY "$WAN_DIR" "hf mirror base download"
elif [ -d "$WAN_DIR" ] && [ -n "$(find "$WAN_DIR" -type f 2>/dev/null | head -1)" ]; then
  record_status wan_base READY_OR_PARTIAL "$WAN_DIR" "hf mirror base partial; inspect inventory"
else
  record_status wan_base BLOCKED "$WAN_DIR" "hf mirror base failed or empty"
fi

DATASET_DIR="$DATASET/Kunbyte_ROSE_Dataset"
mkdir -p "$DATASET_DIR"
run_step dataset_readme curl -fL "$HF_MIRROR/datasets/Kunbyte/ROSE-Dataset/resolve/main/README.md" -o "$DATASET_DIR/README.md" || true
run_step dataset_gitattributes curl -fL "$HF_MIRROR/datasets/Kunbyte/ROSE-Dataset/resolve/main/.gitattributes" -o "$DATASET_DIR/.gitattributes" || true
timeout 180 bash -lc "HF_ENDPOINT=$HF_MIRROR python3 - <<'PY' > '$RUNTIME/rose_dataset_filelist.txt' 2> '$RUNTIME/rose_dataset_filelist.err'
from huggingface_hub import HfApi
api = HfApi(endpoint='https://hf-mirror.com')
for f in api.list_repo_files('Kunbyte/ROSE-Dataset', repo_type='dataset'):
    print(f)
PY" || true
if [ -s "$RUNTIME/rose_dataset_filelist.txt" ]; then
  record_status dataset_filelist READY "$RUNTIME/rose_dataset_filelist.txt" "hf mirror api list_repo_files"
else
  record_status dataset_filelist PARTIAL "$RUNTIME/rose_dataset_filelist.txt" "file list unavailable or timed out"
fi
if [ -s "$DATASET_DIR/README.md" ] || [ -s "$DATASET_DIR/.gitattributes" ]; then
  record_status dataset_metadata READY "$DATASET_DIR" "README/.gitattributes staged only"
else
  record_status dataset_metadata PARTIAL "$DATASET_DIR" "metadata files unavailable"
fi

INV="$REPO_ROOT/reports/exp49_rose_asset_inventory.txt"
SHA="$REPO_ROOT/reports/exp49_rose_asset_sha256.txt"
CSV="$REPO_ROOT/reports/exp49_rose_asset_download.csv"
SUMMARY="$REPO_ROOT/reports/exp49_rose_asset_download_summary.json"
REPORT="$REPO_ROOT/reports/exp49_rose_asset_download.md"

{
  echo "# Exp49 ROSE asset inventory"
  echo "generated=$(date -Ins)"
  echo "hostname=$(hostname)"
  echo
  for d in "$CODE_DIR" "$SPACE_DIR" "$MODEL_DIR" "$WAN_DIR" "$DATASET_DIR"; do
    echo "## $d"
    if [ -e "$d" ]; then
      du -sh "$d" 2>/dev/null || true
      find "$d" -maxdepth 4 -type f | head -300
      count="$(find "$d" -type f 2>/dev/null | wc -l | tr -d ' ')"
      echo "FILE_COUNT $count"
    else
      echo "MISSING"
    fi
    echo
  done
} > "$INV"

{
  echo "# sha256 for Exp49 ROSE assets"
  echo "generated=$(date -Ins)"
  for d in "$CODE_DIR" "$SPACE_DIR" "$MODEL_DIR" "$WAN_DIR" "$DATASET_DIR"; do
    if [ -d "$d" ]; then
      find "$d" -type f -size -128M -print0 2>/dev/null | sort -z | xargs -0 -r sha256sum
      find "$d" -type f -size +128M -printf 'SKIP_LARGE %s %p\n' 2>/dev/null
    fi
  done
} > "$SHA"

python3 - <<PY
import csv, json, os, pathlib, subprocess
runtime = pathlib.Path("$RUNTIME")
rows = []
with open(runtime / "exp49_asset_status.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        rows.append(r)
with open("$CSV", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["asset", "status", "path", "note", "exists", "file_count", "size"])
    w.writeheader()
    for r in rows:
        p = r["path"]
        exists = os.path.exists(p)
        file_count = 0
        size = "NA"
        if exists:
            if os.path.isdir(p):
                for _, _, files in os.walk(p):
                    file_count += len(files)
                try:
                    size = subprocess.check_output(["du", "-sh", p], text=True).split()[0]
                except Exception:
                    size = "NA"
            else:
                file_count = 1
                try:
                    size = str(os.path.getsize(p))
                except Exception:
                    size = "NA"
        rr = dict(r)
        rr.update({"exists": exists, "file_count": file_count, "size": size})
        w.writerow(rr)
summary = {
    "generated": subprocess.getoutput("date -Ins"),
    "hostname": subprocess.getoutput("hostname"),
    "status_rows": rows,
    "ready_assets": [r["asset"] for r in rows if r["status"] == "READY"],
    "blocked_assets": [r["asset"] for r in rows if r["status"] == "BLOCKED"],
    "partial_assets": [r["asset"] for r in rows if r["status"] in ("PARTIAL", "READY_OR_PARTIAL")],
    "policy": "PAI direct download attempted; huggingface.co timed out; hf-mirror.com used; no H20 relay used",
}
statuses = [r["status"] for r in rows]
if any(s == "BLOCKED" for s in statuses):
    final = "ROSE_ASSETS_PARTIAL"
elif any(s in ("PARTIAL", "READY_OR_PARTIAL") for s in statuses):
    final = "ROSE_ASSETS_PARTIAL"
else:
    final = "ROSE_ASSETS_READY"
summary["final_status"] = final
with open("$SUMMARY", "w") as f:
    json.dump(summary, f, indent=2)
PY

final_status="$(python3 - <<PY
import json
print(json.load(open("$SUMMARY"))["final_status"])
PY
)"

cat > "$REPORT" <<MD
# Exp49 ROSE Asset Download

Status: \`$final_status\`

Generated: $(date -Ins)
Host: $(hostname)
Branch: $(git branch --show-current 2>/dev/null || true)
Commit: $(git rev-parse HEAD 2>/dev/null || true)

## Method

PAI direct download was attempted. Direct \`huggingface.co\` calls timed out from PAI, so the PAI download used \`https://hf-mirror.com\`. H20 relay was not used.

## Paths

| Asset | Path |
| --- | --- |
| ROSE code | \`$CODE_DIR\` |
| HF Space | \`$SPACE_DIR\` |
| HF model | \`$MODEL_DIR\` |
| Wan base model | \`$WAN_DIR\` |
| ROSE dataset metadata | \`$DATASET_DIR\` |

## Asset Status

\`\`\`text
$(cat "$RUNTIME/exp49_asset_status.tsv")
\`\`\`

## Inventory And Checksums

- Inventory: \`reports/exp49_rose_asset_inventory.txt\`
- SHA256: \`reports/exp49_rose_asset_sha256.txt\`
- Summary JSON: \`reports/exp49_rose_asset_download_summary.json\`

Files over 128 MiB are listed as \`SKIP_LARGE\` in the checksum report to avoid blocking the milestone on multi-hour NAS hashing.

## Disk Snapshot

\`\`\`text
$(df -h /mnt/nas /mnt/workspace "$BASE" 2>/dev/null || true)
\`\`\`

## Notes

No raw assets were added to Git. No GPU work, inference, training, or optimizer step was run in this milestone.
MD

cat > experiment_registry/exp49_pai_rose_adapter_feasibility/status.md <<MD
# Exp49 Status

Current status: \`$final_status\`

Milestone B attempted PAI direct download for ROSE code, HF Space, ROSE model, Wan base model, and ROSE-Dataset metadata/sample inventory. Direct HuggingFace timed out, so PAI used hf-mirror. No H20 relay, GPU work, inference, or training was used.

Next required action: run Milestone C environment/import smoke if code/model/base assets are sufficient.
MD

python3 - <<PY
from pathlib import Path
p = Path("experiment_registry/exp49_pai_rose_adapter_feasibility/results.tsv")
lines = p.read_text().splitlines()
out = []
for line in lines:
    if line.startswith("B_assets\t"):
        out.append("B_assets\t$final_status\tPAI direct asset download used hf-mirror after huggingface.co timeout; see reports/exp49_rose_asset_download.md.")
    else:
        out.append(line)
p.write_text("\n".join(out) + "\n")
PY

cat > experiment_registry/exp49_pai_rose_adapter_feasibility/metric_summary.md <<MD
# Exp49 Metric Summary

Milestone B generated asset inventory and checksums only. No ROSE inference metrics have been computed yet.

Asset status: \`$final_status\`.
MD

cat > experiment_registry/exp49_pai_rose_adapter_feasibility/qualitative_summary.md <<MD
# Exp49 Qualitative Summary

No ROSE visual evidence has been generated or reviewed yet.

Milestone B only downloaded/audited assets and produced inventory/checksum reports.
MD

EXP49_HOST="$(hostname)" EXP49_STATUS="$final_status" python3 - <<'PY'
from pathlib import Path
import os
hostname = os.environ["EXP49_HOST"]
status = os.environ["EXP49_STATUS"]
updates = {
 "PRD/00_current_status.md": f"""

## 2026-06-30 Exp49 ROSE Asset Download

Exp49 Milestone B was executed on verified PAI host `{hostname}`. Direct `huggingface.co` access timed out, so PAI direct download used `hf-mirror.com`. Status: `{status}`. Asset inventory, checksum report, CSV, and summary JSON were written under `reports/exp49_rose_asset_*`. No H20 relay, GPU work, inference, training, or optimizer step was used.
""",
 "PRD/01_experiment_matrix.md": f"""

## 2026-06-30 Exp49 Asset Download Update

| Experiment | Milestone | Status | Notes |
| --- | --- | --- | --- |
| `exp49_pai_rose_adapter_feasibility` | B assets | `{status}` | PAI direct download via hf-mirror after huggingface.co timeout; see `reports/exp49_rose_asset_download.md`; no training or inference run. |
""",
 "PRD/46_exp49_pai_rose_adapter_feasibility.md": f"""

## Milestone B Update - 2026-06-30

Status: `{status}`.

PAI direct download was attempted for ROSE official code, HF Space, ROSE model, Wan base model, and ROSE-Dataset metadata/filelist. Direct `huggingface.co` timed out from PAI, so `hf-mirror.com` was used. Reports are available at `reports/exp49_rose_asset_download.md`, `reports/exp49_rose_asset_download.csv`, `reports/exp49_rose_asset_download_summary.json`, `reports/exp49_rose_asset_inventory.txt`, and `reports/exp49_rose_asset_sha256.txt`.

No H20 relay, GPU work, inference, training, or optimizer step was used.
""",
}
for name, text in updates.items():
    p = Path(name)
    cur = p.read_text()
    marker = text.strip().split("\n", 1)[0]
    if marker in cur:
        before = cur.split(marker, 1)[0].rstrip()
        p.write_text(before + "\n\n" + text.strip() + "\n")
    else:
        p.write_text(cur.rstrip() + text + "\n")
PY

printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
  "$(date -Ins)" "$(hostname)" "$(git branch --show-current 2>/dev/null || true)" "$(git rev-parse HEAD 2>/dev/null || true)" \
  "B_assets_done" "none" "$$" "$(ps -o pgid= $$ | tr -d ' ')" "NA" "NA" "assets" "0" "NA" "NA" "none" "none" "$final_status" "none" "commit milestone B" >> "$MONITOR"

echo
cat "$REPORT"
echo
echo "## git status after Milestone B"
git status --short
