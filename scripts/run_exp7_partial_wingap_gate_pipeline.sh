#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

export PROJECT_ROOT
export OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
export EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-/mnt/nas/hj/H20_Video_inpainting_DPO/experiments}"
export WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/nas/hj/weights}"
export CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/mnt/nas/hj/conda_envs/diffueraser}"
export CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_PREFIX}}"
export VBENCH_CONDA_ENV="${VBENCH_CONDA_ENV:-/mnt/nas/hj/conda_envs/videodpo}"

export EXP_NAME="${EXP_NAME:-exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500}"
export PREFERENCE_MANIFEST="${PREFERENCE_MANIFEST:-/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4/manifests/selected_primary_comp.repaired.jsonl}"
export TRAIN_MASK_MODE="${TRAIN_MASK_MODE:-partial}"
export MASK_FROM_MANIFEST="${MASK_FROM_MANIFEST:-true}"
export LOSS_REGION_MODE="${LOSS_REGION_MODE:-full}"

export BETA_DPO="${BETA_DPO:-10}"
export LOSE_GAP_WEIGHT="${LOSE_GAP_WEIGHT:-0.25}"
export DPO_LOSE_GAP_WEIGHT="${DPO_LOSE_GAP_WEIGHT:-${LOSE_GAP_WEIGHT}}"
export WINNER_ABS_REG_WEIGHT="${WINNER_ABS_REG_WEIGHT:-0.05}"
export WINNER_GAP_REG_WEIGHT="${WINNER_GAP_REG_WEIGHT:-1.0}"
export WINNER_GAP_REG_MARGIN="${WINNER_GAP_REG_MARGIN:-0.0}"
export WINNER_GAP_REG_MODE="${WINNER_GAP_REG_MODE:-relu}"
export SFT_REG_WEIGHT="${SFT_REG_WEIGHT:-0.0}"

export STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-1500}"
export STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-1500}"
export NUM_GPUS="${NUM_GPUS:-8}"
export VAL_STEPS="${VAL_STEPS:-999999}"
export CKPT_STEPS="${CKPT_STEPS:-500}"
export CKPT_LIMIT="${CKPT_LIMIT:-4}"
export REPORT_TO="${REPORT_TO:-none}"
export DPO_DIAG_SAVE_WANDB="${DPO_DIAG_SAVE_WANDB:-false}"
export ENABLE_DPO_DIAG="${ENABLE_DPO_DIAG:-true}"
export DPO_DIAG_SAVE_CSV="${DPO_DIAG_SAVE_CSV:-true}"
export DPO_DIAG_LOG_EVERY="${DPO_DIAG_LOG_EVERY:-10}"
export LINGBOT_PROCESS_NAME="${LINGBOT_PROCESS_NAME:-lingbot-worldphy}"
export TRAIN_HEIGHT="${TRAIN_HEIGHT:-320}"
export TRAIN_WIDTH="${TRAIN_WIDTH:-512}"
export RESOLUTION="${RESOLUTION:-512}"
export NFRAMES="${NFRAMES:-16}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/external/VideoDPO/prompts/vbench_standard_prompts.txt}"
export QUAL30_SEED="${QUAL30_SEED:-42}"
export RUN_QUAL30="${RUN_QUAL30:-true}"
export RUN_FULL_VBENCH="${RUN_FULL_VBENCH:-false}"

if [[ "${RUN_QUAL30,,}" == "true" || "${RUN_QUAL30}" == "1" || "${RUN_QUAL30,,}" == "yes" ]]; then
  export SKIP_QUAL30=false
else
  export SKIP_QUAL30=true
fi

if [[ "${RUN_FULL_VBENCH,,}" == "true" || "${RUN_FULL_VBENCH}" == "1" || "${RUN_FULL_VBENCH,,}" == "yes" ]]; then
  export SKIP_FULL_VBENCH=false
else
  export SKIP_FULL_VBENCH=true
fi

export PIPELINE_TS="${PIPELINE_TS:-$(date +%Y%m%d_%H%M%S)}"
export RUN_VERSION="${RUN_VERSION:-${PIPELINE_TS}}"
PIPELINE_ROOT="${OUTPUT_ROOT}/logs/pipelines/${EXP_NAME}"
PIPELINE_TS_ROOT="${PIPELINE_ROOT}/${PIPELINE_TS}"
REPORT_DIR="${OUTPUT_ROOT}/reports"
mkdir -p "${PIPELINE_ROOT}" "${REPORT_DIR}"

die() {
  echo "[EXP7-GATE][ERROR] $*" >&2
  exit 1
}

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "$path" ]] || die "${label} not found: ${path}"
}

resolve_python() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    echo "${PYTHON_BIN}"
  elif [[ -d "${CONDA_ENV}" && -x "${CONDA_ENV}/bin/python" ]]; then
    echo "${CONDA_ENV}/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    command -v python3
  elif command -v python >/dev/null 2>&1; then
    command -v python
  else
    die "python not found; set PYTHON_BIN or CONDA_ENV/CONDA_ENV_PREFIX"
  fi
}

PYTHON_BIN="$(resolve_python)"
export PYTHON_BIN

echo "[EXP7-GATE] precheck"
require_path "${PREFERENCE_MANIFEST}" "PREFERENCE_MANIFEST"
require_path "${WEIGHTS_DIR}/stable-diffusion-v1-5" "stable-diffusion-v1-5"
require_path "${WEIGHTS_DIR}/sd-vae-ft-mse" "sd-vae-ft-mse"
require_path "${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000" "DiffuEraser-base converted weights"
require_path "${PROJECT_ROOT}/training/dpo/scripts/03_dpo_stage1.sbatch" "Stage1 launcher"
require_path "${PROJECT_ROOT}/training/dpo/scripts/04_dpo_stage2.sbatch" "Stage2 launcher"
require_path "${PROJECT_ROOT}/training/dpo/train_stage2.py" "train_stage2.py"
require_path "${PROJECT_ROOT}/DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh" "qual30 runner"
require_path "${PROMPTS_FILE}" "VBench prompt file"

case "${TRAIN_MASK_MODE}:${MASK_FROM_MANIFEST}" in
  partial:true|partial:True|partial:1) ;;
  *) die "Exp7 gate requires TRAIN_MASK_MODE=partial and MASK_FROM_MANIFEST=true; got ${TRAIN_MASK_MODE}/${MASK_FROM_MANIFEST}" ;;
esac

export PIPELINE_ROOT PIPELINE_TS_ROOT REPORT_DIR
"${PYTHON_BIN}" - <<'PY'
import json
import os
import random
from pathlib import Path

manifest = Path(os.environ["PREFERENCE_MANIFEST"])
rows = []
with manifest.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
if len(rows) != 10000:
    raise SystemExit(f"manifest row count must be 10000 for Exp7 gate, got {len(rows)}")

required = ["win_video_path", "final_loser_video_path", "mask_path"]
sample = random.Random(42).sample(rows, min(8, len(rows)))
for i, row in enumerate(sample):
    for key in required:
        value = row.get(key)
        if not value:
            raise SystemExit(f"sample {i} missing {key}")
        path = Path(value)
        if not path.exists():
            raise SystemExit(f"sample {i} {key} not found: {path}")
        if not os.access(path, os.R_OK):
            raise SystemExit(f"sample {i} {key} not readable: {path}")

config_keys = [
    "EXP_NAME",
    "PREFERENCE_MANIFEST",
    "TRAIN_MASK_MODE",
    "MASK_FROM_MANIFEST",
    "LOSS_REGION_MODE",
    "BETA_DPO",
    "LOSE_GAP_WEIGHT",
    "WINNER_ABS_REG_WEIGHT",
    "WINNER_GAP_REG_WEIGHT",
    "WINNER_GAP_REG_MARGIN",
    "WINNER_GAP_REG_MODE",
    "SFT_REG_WEIGHT",
    "STAGE1_MAX_STEPS",
    "STAGE2_MAX_STEPS",
    "CKPT_STEPS",
    "CKPT_LIMIT",
    "RUN_QUAL30",
    "RUN_FULL_VBENCH",
    "WEIGHTS_DIR",
    "EXPERIMENTS_DIR",
    "NUM_GPUS",
    "PIPELINE_TS",
    "RUN_VERSION",
]
manifest_json = {key: os.environ.get(key, "") for key in config_keys}
manifest_json["manifest_rows"] = len(rows)
manifest_json["sampled_paths_checked"] = len(sample)
root = Path(os.environ["PIPELINE_ROOT"])
ts_root = Path(os.environ["PIPELINE_TS_ROOT"])
root.mkdir(parents=True, exist_ok=True)
ts_root.mkdir(parents=True, exist_ok=True)
(root / "pipeline_manifest.json").write_text(json.dumps(manifest_json, indent=2) + "\n", encoding="utf-8")
(ts_root / "pipeline_manifest.json").write_text(json.dumps(manifest_json, indent=2) + "\n", encoding="utf-8")
print(f"[EXP7-GATE] manifest rows={len(rows)} sampled={len(sample)}")
PY

bash "${PROJECT_ROOT}/scripts/run_dpo_two_stage_vbench_pipeline.sh"

STAGE1_RUN_DIR="${EXPERIMENTS_DIR}/dpo/stage1/${RUN_VERSION}_${EXP_NAME}_stage1"
STAGE2_RUN_DIR="${EXPERIMENTS_DIR}/dpo/stage2/${RUN_VERSION}_${EXP_NAME}_stage2"
QUAL_ROOT="${OUTPUT_ROOT}/logs/qual_sbs_30/${EXP_NAME}_${PIPELINE_TS}"
SUMMARY_PATH="${REPORT_DIR}/${EXP_NAME}_dpo_diag_summary.md"
GATE_REPORT="${REPORT_DIR}/${EXP_NAME}_gate_report.md"

require_path "${STAGE1_RUN_DIR}/last_weights" "Stage1 last_weights"
require_path "${STAGE2_RUN_DIR}/last_weights" "Stage2 last_weights"
require_path "${STAGE1_RUN_DIR}/dpo_diagnostics.csv" "Stage1 dpo_diagnostics.csv"
require_path "${STAGE2_RUN_DIR}/dpo_diagnostics.csv" "Stage2 dpo_diagnostics.csv"

export STAGE1_RUN_DIR STAGE2_RUN_DIR SUMMARY_PATH GATE_REPORT QUAL_ROOT
"${PYTHON_BIN}" - <<'PY'
import csv
import math
import os
import statistics
from pathlib import Path

metrics = [
    "dpo_loss",
    "implicit_acc",
    "mse_w",
    "ref_mse_w",
    "mse_l",
    "ref_mse_l",
    "win_gap",
    "lose_gap",
    "winner_abs_reg",
    "winner_gap_reg",
    "mse_w_over_ref_mse_w",
    "mse_l_over_ref_mse_l",
    "sigma_term",
    "kl_divergence",
    "grad_norm",
]
required_cols = {
    "winner_abs_reg",
    "winner_gap_reg",
    "lose_gap_weight",
    "mse_w_over_ref_mse_w",
    "mse_l_over_ref_mse_l",
    "win_gap_positive_ratio",
}

def parse_float(value):
    try:
        if value in {"", None}:
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None

def p90(values):
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, math.ceil(0.9 * len(values)) - 1)
    return values[idx]

def fmt(value):
    if value is None:
        return "NA"
    return f"{value:.6g}"

def summarize(stage, csv_path):
    path = Path(csv_path)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"{stage} diagnostics CSV is empty: {path}")
    missing = sorted(required_cols - set(rows[0].keys()))
    if missing:
        raise SystemExit(f"{stage} diagnostics missing columns: {missing}")

    stats = {}
    for metric in metrics:
        values = [parse_float(row.get(metric)) for row in rows]
        values = [v for v in values if v is not None]
        if values:
            stats[metric] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "p90": p90(values),
                "max": max(values),
            }
    fractions = {
        "frac_dpo_loss_lt_1e-3": ("dpo_loss", lambda v: v < 1e-3),
        "frac_implicit_acc_gt_0.99": ("implicit_acc", lambda v: v > 0.99),
        "frac_mse_w_over_ref_gt_5": ("mse_w_over_ref_mse_w", lambda v: v > 5),
        "frac_win_gap_gt_0.5": ("win_gap", lambda v: v > 0.5),
        "frac_sigma_term_gt_0.99": ("sigma_term", lambda v: v > 0.99),
        "frac_kl_divergence_gt_1": ("kl_divergence", lambda v: v > 1.0),
    }
    frac_stats = {}
    for name, (metric, pred) in fractions.items():
        values = [parse_float(row.get(metric)) for row in rows]
        values = [v for v in values if v is not None]
        frac_stats[name] = sum(1 for v in values if pred(v)) / len(values) if values else None
    return rows, stats, frac_stats

stage_paths = {
    "Stage1": Path(os.environ["STAGE1_RUN_DIR"]) / "dpo_diagnostics.csv",
    "Stage2": Path(os.environ["STAGE2_RUN_DIR"]) / "dpo_diagnostics.csv",
}
all_summaries = {stage: summarize(stage, path) for stage, path in stage_paths.items()}

stage2_frac = all_summaries["Stage2"][2]
fail_flags = [
    (stage2_frac["frac_mse_w_over_ref_gt_5"] or 0) > 0.2,
    (stage2_frac["frac_win_gap_gt_0.5"] or 0) > 0.2,
    (stage2_frac["frac_sigma_term_gt_0.99"] or 0) > 0.5,
    (stage2_frac["frac_kl_divergence_gt_1"] or 0) > 0.5,
]
risky_flags = [
    (stage2_frac["frac_mse_w_over_ref_gt_5"] or 0) > 0.05,
    (stage2_frac["frac_win_gap_gt_0.5"] or 0) > 0.05,
    (stage2_frac["frac_sigma_term_gt_0.99"] or 0) > 0.2,
    (stage2_frac["frac_kl_divergence_gt_1"] or 0) > 0.2,
]
verdict = "FAIL_LIKELY" if any(fail_flags) else "RISKY" if any(risky_flags) else "PASS_LIKELY"

out = Path(os.environ["SUMMARY_PATH"])
with out.open("w", encoding="utf-8") as f:
    f.write("# Exp7 Gate DPO Diagnostics Summary\n\n")
    f.write(f"verdict: **{verdict}**\n\n")
    for stage, (rows, stats, frac_stats) in all_summaries.items():
        f.write(f"## {stage}\n\n")
        f.write(f"source: `{stage_paths[stage]}`\n\n")
        f.write(f"rows: {len(rows)}\n\n")
        f.write("| metric | mean | median | p90 | max |\n")
        f.write("| --- | ---: | ---: | ---: | ---: |\n")
        for metric in metrics:
            if metric not in stats:
                continue
            s = stats[metric]
            f.write(f"| {metric} | {fmt(s['mean'])} | {fmt(s['median'])} | {fmt(s['p90'])} | {fmt(s['max'])} |\n")
        f.write("\n")
        f.write("| gate fraction | value |\n")
        f.write("| --- | ---: |\n")
        for key, value in frac_stats.items():
            f.write(f"| {key} | {fmt(value)} |\n")
        f.write("\n")
print(out)
print(verdict)
PY

if [[ "${SKIP_QUAL30}" == "false" ]]; then
  require_path "${QUAL_ROOT}/pair_manifest.csv" "qual30 pair_manifest.csv"
  require_path "${QUAL_ROOT}/index.html" "qual30 index.html"
  require_path "${QUAL_ROOT}/side_by_side" "qual30 side_by_side dir"
fi

{
  echo "# ${EXP_NAME} Gate Report"
  echo
  echo "status: launched/completed gate pipeline"
  echo "stage1_run_dir: \`${STAGE1_RUN_DIR}\`"
  echo "stage2_run_dir: \`${STAGE2_RUN_DIR}\`"
  echo "stage1_log: \`${OUTPUT_ROOT}/logs/train/${EXP_NAME}/stage1.log\`"
  echo "stage2_log: \`${OUTPUT_ROOT}/logs/train/${EXP_NAME}/stage2.log\`"
  echo "train_mask_mode: ${TRAIN_MASK_MODE}"
  echo "mask_from_manifest: ${MASK_FROM_MANIFEST}"
  echo "loss_region_mode: ${LOSS_REGION_MODE}"
  echo "beta_dpo: ${BETA_DPO}"
  echo "lose_gap_weight: ${LOSE_GAP_WEIGHT}"
  echo "winner_abs_reg_weight: ${WINNER_ABS_REG_WEIGHT}"
  echo "winner_gap_reg_weight: ${WINNER_GAP_REG_WEIGHT}"
  echo "stage1_steps: ${STAGE1_MAX_STEPS}"
  echo "stage2_steps: ${STAGE2_MAX_STEPS}"
  echo "qual30_side_by_side_dir: \`${QUAL_ROOT}/side_by_side\`"
  echo "qual30_index: \`${QUAL_ROOT}/index.html\`"
  echo "dpo_diag_summary: \`${SUMMARY_PATH}\`"
  echo "full_vbench_default: disabled"
} > "${GATE_REPORT}"

echo "[EXP7-GATE] gate report: ${GATE_REPORT}"
echo "[EXP7-GATE] dpo summary: ${SUMMARY_PATH}"
