#!/usr/bin/env bash
set -euo pipefail

# Exp31 VideoPainter 2000-step evaluation controller.
# This launches evaluation only: no training, no MiniMax, no adapter updates.

PROJECT_ROOT="${PROJECT_ROOT:-/home/hj/runtime_code/H20_Video_inpainting_DPO_exp31_vp2000}"
CODE_DIR="$PROJECT_ROOT/exp26_videopainter_dpo_v2/code"
REPORTS_DIR="$PROJECT_ROOT/reports"
LOG_ROOT="${LOG_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp31_videopainter_2000step_longrun}"
TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-$LOG_ROOT/exp31_vp2000_fresh_step0_20260627_133831}"
RUN_ID="${RUN_ID:-exp31_vp2000_eval_step0_50_2000_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$LOG_ROOT/$RUN_ID}"
RUNTIME_DIR="${RUNTIME_DIR:-/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp31_vp2000_eval_step0_50_2000}"
GPU="${GPU:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VP_ROOT="${VP_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter}"
VP_BASE="${VP_BASE:-$VP_ROOT/ckpt/CogVideoX-5b-I2V}"
SEARCH_MASK_READY="${SEARCH_MASK_READY:-/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/searchdev_step0_official_20260625_131957/gate64_mask_ready.jsonl}"
SHADOW_MASK_READY="${SHADOW_MASK_READY:-/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/shadowdev_confirmatory_20260625/gate64_mask_ready.jsonl}"
STEPS="${STEPS:-0 50 2000}"

mkdir -p "$RUN_ROOT" "$RUNTIME_DIR" "$REPORTS_DIR"

log() {
  printf '[%s] %s\n' "$(date -Ins)" "$*" | tee -a "$RUN_ROOT/controller.log"
}

fail() {
  log "ERROR: $*"
  echo "FAILED: $*" > "$RUNTIME_DIR/status"
  exit 2
}

require_safe_gpu() {
  if [[ "$GPU" == "0" ]]; then
    fail "GPU0 is reserved for the right plugin and cannot be used"
  fi
  local uuid
  uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F',' -v i="$GPU" '$1+0==i+0 {gsub(/ /,"",$2); print $2}')"
  [[ -n "$uuid" ]] || fail "GPU$GPU not found"
  if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null | grep -q "$uuid"; then
    fail "GPU$GPU has an existing compute process"
  fi
}

checkpoint_dir() {
  local step="$1"
  echo "$TRAIN_RUN_ROOT/checkpoint-$step"
}

checkpoint_audit() {
  local csv="$RUN_ROOT/exp31_vp_2000_checkpoint_audit.csv"
  local md="$REPORTS_DIR/exp31_vp_2000_checkpoint_audit.md"
  local json="$REPORTS_DIR/exp31_vp_2000_checkpoint_audit.json"
  echo "step,checkpoint_dir,branch_dir,config_exists,weights_exists,trainer_state_exists,weights_bytes,weights_sha256,strict_load_status" > "$csv"
  local rows_json="$RUN_ROOT/exp31_vp_2000_checkpoint_audit.rows.jsonl"
  : > "$rows_json"
  for step in 0 1 10 50 100 200 500 1000 1500 2000; do
    local ckpt branch config weights trainer bytes sha status
    ckpt="$(checkpoint_dir "$step")"
    branch="$ckpt/branch"
    config="$branch/config.json"
    weights="$branch/diffusion_pytorch_model.safetensors"
    trainer="$ckpt/trainer_state.pt"
    bytes=0
    sha=""
    status="pending_generation_runtime_strict_load"
    if [[ -s "$weights" ]]; then
      bytes="$(stat -c '%s' "$weights")"
      sha="$(sha256sum "$weights" | awk '{print $1}')"
    else
      status="missing_weights"
    fi
    printf '%s,"%s","%s",%s,%s,%s,%s,%s,%s\n' \
      "$step" "$ckpt" "$branch" \
      "$([[ -s "$config" ]] && echo true || echo false)" \
      "$([[ -s "$weights" ]] && echo true || echo false)" \
      "$([[ -s "$trainer" ]] && echo true || echo false)" \
      "$bytes" "$sha" "$status" >> "$csv"
    "$PYTHON_BIN" - "$step" "$ckpt" "$branch" "$config" "$weights" "$trainer" "$bytes" "$sha" "$status" >> "$rows_json" <<'PY'
import json
import sys
step, ckpt, branch, config, weights, trainer, bytes_, sha, status = sys.argv[1:]
print(json.dumps({
    "step": int(step),
    "checkpoint_dir": ckpt,
    "branch_dir": branch,
    "config_exists": bool(config and __import__("pathlib").Path(config).is_file()),
    "weights_exists": bool(weights and __import__("pathlib").Path(weights).is_file()),
    "trainer_state_exists": bool(trainer and __import__("pathlib").Path(trainer).is_file()),
    "weights_bytes": int(bytes_),
    "weights_sha256": sha,
    "strict_load_status": status,
}, sort_keys=True))
PY
  done
  "$PYTHON_BIN" - "$rows_json" "$json" <<'PY'
import json
import sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
summary = {
    "status": "VIDEOPAINTER_2000_CHECKPOINTS_READY" if all(r["config_exists"] and r["weights_exists"] for r in rows) else "VIDEOPAINTER_2000_EVAL_BLOCKED_CHECKPOINT_LOAD",
    "rows": rows,
}
open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
  {
    echo "# Exp31 VideoPainter 2000-step Checkpoint Audit"
    echo
    echo "- run_root: \`$TRAIN_RUN_ROOT\`"
    echo "- audit_csv: \`$csv\`"
    echo "- strict_load_status: runtime strict load is verified when each generation job instantiates the branch."
    echo
    echo "| step | config | weights | trainer_state | weights_bytes | sha256 |"
    echo "| --- | --- | --- | --- | --- | --- |"
    tail -n +2 "$csv" | while IFS=',' read -r step ckpt branch config weights trainer bytes sha status; do
      printf '| %s | %s | %s | %s | %s | `%s` |\n' "$step" "$config" "$weights" "$trainer" "$bytes" "$sha"
    done
  } > "$md"
  cp "$csv" "$REPORTS_DIR/exp31_vp_2000_checkpoint_audit.csv"
  [[ "$(python - <<PY
import json
print(json.load(open("$json"))["status"])
PY
)" == "VIDEOPAINTER_2000_CHECKPOINTS_READY" ]] || fail "checkpoint audit failed"
}

prepare_split_manifest() {
  local split="$1"
  local src="$2"
  local dst="$RUN_ROOT/$split/gate64_mask_ready.jsonl"
  [[ -s "$src" ]] || fail "missing mask-ready manifest for $split: $src"
  mkdir -p "$RUN_ROOT/$split"
  if [[ ! -e "$dst" ]]; then
    ln -s "$src" "$dst"
  fi
  local n
  n="$(wc -l < "$dst" | tr -d ' ')"
  [[ "$n" == "32" ]] || fail "$split manifest has $n rows, expected 32"
}

run_one() {
  local split="$1"
  local step="$2"
  local manifest="$RUN_ROOT/$split/gate64_mask_ready.jsonl"
  local ckpt
  ckpt="$(checkpoint_dir "$step")"
  local step_root="$RUN_ROOT/$split/step$step"
  local out="$step_root/official_generation"
  local review="$step_root/step${step}_review"
  local step_log="$RUN_ROOT/${split}_step${step}.log"
  mkdir -p "$step_root"
  if [[ -s "$out/gate64_generation_summary.json" ]] && grep -q '"status": "passed"' "$out/gate64_generation_summary.json"; then
    log "$split step$step generation already passed; refreshing review"
  else
    log "start $split step$step on GPU$GPU"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$CODE_DIR/run_vp2_gate64_official_generation.py" \
      --videopainter-root "$VP_ROOT" \
      --base-model "$VP_BASE" \
      --branch-checkpoint "$ckpt" \
      --manifest "$manifest" \
      --output-dir "$out" \
      --limit 32 \
      --height 480 \
      --width 720 \
      --num-frames 49 \
      --num-inference-steps 20 \
      --guidance-scale 6.0 \
      --seed 20260627 \
      --dtype bf16 \
      --device cuda \
      --skip-existing 2>&1 | tee "$step_log"
  fi
  "$PYTHON_BIN" "$CODE_DIR/review_gate64_official_outputs.py" \
    --run-root "$step_root" \
    --manifest "$manifest" \
    --output-dir "$review" \
    --num-frames 49 | tee -a "$step_log"
  log "complete $split step$step"
}

write_eval_summary() {
  "$PYTHON_BIN" - "$RUN_ROOT" "$REPORTS_DIR" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
reports_dir = Path(sys.argv[2])
rows = []
for split in ("search", "shadow"):
    for step in (0, 50, 2000):
        review_csv = run_root / split / f"step{step}" / f"step{step}_review" / "gate64_visual_review.csv"
        gen_json = run_root / split / f"step{step}" / "official_generation" / "gate64_generation_summary.json"
        data = []
        if review_csv.exists():
            with review_csv.open(encoding="utf-8", newline="") as handle:
                data = list(csv.DictReader(handle))
        rows.append({
            "split": split,
            "step": step,
            "generation_summary": str(gen_json),
            "review_csv": str(review_csv),
            "rows": len(data),
            "ok_rows": sum(1 for r in data if r.get("status") == "OK"),
            "full_psnr_mean": float(sum(float(r["full_psnr"]) for r in data if r.get("full_psnr")) / max(1, sum(1 for r in data if r.get("full_psnr")))),
            "mask_psnr_mean": float(sum(float(r["mask_psnr"]) for r in data if r.get("mask_psnr")) / max(1, sum(1 for r in data if r.get("mask_psnr")))),
            "full_ssim_mean": float(sum(float(r["full_ssim"]) for r in data if r.get("full_ssim")) / max(1, sum(1 for r in data if r.get("full_ssim")))),
            "technical_invalid": sum(1 for r in data if r.get("classification") == "technical-invalid"),
            "medium_hard": sum(1 for r in data if r.get("classification") == "medium-hard"),
            "hard_plausible": sum(1 for r in data if r.get("classification") == "hard-plausible"),
            "trivial_bad": sum(1 for r in data if r.get("classification") == "trivial-bad"),
        })
reports_dir.mkdir(parents=True, exist_ok=True)
csv_path = reports_dir / "exp31_vp_2000_step0_50_2000_eval_summary.csv"
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
summary = {
    "status": "VIDEOPAINTER_2000_STEP0_50_2000_EVALUATED" if all(r["rows"] == 32 and r["ok_rows"] == 32 for r in rows) else "VIDEOPAINTER_2000_STEP0_50_2000_INCOMPLETE",
    "run_root": str(run_root),
    "rows": rows,
}
(reports_dir / "exp31_vp_2000_step0_50_2000_eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
md = ["# Exp31 VideoPainter Step0/50/2000 Evaluation Summary", "", f"Status: `{summary['status']}`", "", f"- run_root: `{run_root}`", f"- csv: `{csv_path}`", "", "| split | step | rows | ok | full_psnr | mask_psnr | medium-hard | hard-plausible | trivial-bad | invalid |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for r in rows:
    md.append(f"| {r['split']} | {r['step']} | {r['rows']} | {r['ok_rows']} | {r['full_psnr_mean']:.4f} | {r['mask_psnr_mean']:.4f} | {r['medium_hard']} | {r['hard_plausible']} | {r['trivial_bad']} | {r['technical_invalid']} |")
md.extend(["", "This report evaluates completed checkpoints only. It does not start new VideoPainter training or claim final SOTA."])
(reports_dir / "exp31_vp_2000_step0_50_2000_eval_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

main() {
  (
    flock -n 9 || fail "lock busy: $RUNTIME_DIR/controller.lock"
    echo "$$" > "$RUNTIME_DIR/pid"
    echo "$(ps -o pgid= -p $$ | tr -d ' ')" > "$RUNTIME_DIR/pgid"
    echo "$GPU" > "$RUNTIME_DIR/gpu"
    echo "$RUN_ROOT" > "$RUNTIME_DIR/run_root"
    echo "RUNNING" > "$RUNTIME_DIR/status"
    log "Exp31 eval controller start RUN_ROOT=$RUN_ROOT GPU=$GPU"
    require_safe_gpu
    checkpoint_audit
    prepare_split_manifest search "$SEARCH_MASK_READY"
    prepare_split_manifest shadow "$SHADOW_MASK_READY"
    for split in search shadow; do
      for step in $STEPS; do
        date -Ins > "$RUNTIME_DIR/heartbeat"
        echo "$split step$step" > "$RUNTIME_DIR/current_task"
        run_one "$split" "$step"
      done
    done
    write_eval_summary
    echo "COMPLETED" > "$RUNTIME_DIR/status"
    log "Exp31 eval controller completed"
  ) 9>"$RUNTIME_DIR/controller.lock"
}

main "$@"
