#!/bin/bash
set -euo pipefail

# Target-domain preflight/eval wrapper.
# This script uses the existing project metric backend through
# tools/run_inpainting_metric_eval.py. It does not run VBench.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/nas/hj/weights}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
TARGET_EVAL_ROOT="${TARGET_EVAL_ROOT:-${OUTPUT_ROOT}/logs/target_eval}"
YOUTUBE_OUT="${YOUTUBE_OUT:-${TARGET_EVAL_ROOT}/youtubevos/${RUN_TS}}"
DAVIS_OUT="${DAVIS_OUT:-${TARGET_EVAL_ROOT}/davis/${RUN_TS}}"
REPORT="${REPORT:-${OUTPUT_ROOT}/reports/target_domain_youtubevos_davis_eval_report.md}"

PAIR_MANIFEST_YOUTUBE="${PAIR_MANIFEST_YOUTUBE:-}"
PAIR_MANIFEST_DAVIS="${PAIR_MANIFEST_DAVIS:-}"
MAX_FRAMES="${MAX_FRAMES:-16}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-320}"
COMPUTE_EWARP="${COMPUTE_EWARP:-false}"
COMPUTE_LPIPS="${COMPUTE_LPIPS:-false}"

mkdir -p "${YOUTUBE_OUT}" "${DAVIS_OUT}" "$(dirname "${REPORT}")"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

flag_args=()
case "${COMPUTE_EWARP,,}" in
  1|true|yes|on) flag_args+=(--compute_ewarp) ;;
esac
case "${COMPUTE_LPIPS,,}" in
  1|true|yes|on) flag_args+=(--compute_lpips) ;;
esac

write_inventory() {
  {
    echo "# Target YouTube-VOS / DAVIS Metric Eval"
    echo
    echo "status: preflight"
    echo
    echo "metric_backend: \`inference/metrics.py\`"
    echo "metric_wrapper: \`tools/run_inpainting_metric_eval.py\`"
    echo "youtubevos_output: \`${YOUTUBE_OUT}\`"
    echo "davis_output: \`${DAVIS_OUT}\`"
    echo
    echo "## Checkpoints"
    echo
    echo "| label | path | status |"
    echo "| --- | --- | --- |"
    while IFS="|" read -r label path; do
      [[ -z "${label}" ]] && continue
      if [[ -e "${path}" ]]; then
        status="present"
      else
        status="missing"
      fi
      echo "| \`${label}\` | \`${path}\` | ${status} |"
    done <<EOF
DiffuEraser-base|${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000
Exp7_DPO_Stage1_last|${OUTPUT_ROOT}/experiments/dpo/stage1/20260601_065618_exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500_stage1/last_weights
Exp7_DPO_S1_DPO_S2_last|${OUTPUT_ROOT}/experiments/dpo/stage2/20260601_065618_exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500_stage2/last_weights
Exp6_DPO_S1_DPO_S2_last|${OUTPUT_ROOT}/experiments/dpo/stage2/20260601_004753_exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000_stage2/last_weights
EOF
    echo
    echo "## Pair Manifests"
    echo
    echo "| domain | manifest | status |"
    echo "| --- | --- | --- |"
    for item in "youtubevos|${PAIR_MANIFEST_YOUTUBE}" "davis|${PAIR_MANIFEST_DAVIS}"; do
      domain="${item%%|*}"
      manifest="${item#*|}"
      if [[ -n "${manifest}" && -f "${manifest}" ]]; then
        status="present"
      elif [[ -n "${manifest}" ]]; then
        status="missing"
      else
        status="not set"
      fi
      echo "| ${domain} | \`${manifest}\` | ${status} |"
    done
    echo
    echo "VBench is intentionally not used here; this is partial-mask inpainting evaluation."
  } > "${REPORT}"
}

run_domain() {
  local domain="$1"
  local manifest="$2"
  local out_dir="$3"
  mkdir -p "${out_dir}/metrics" "${out_dir}/qualitative" "${out_dir}/side_by_side"
  if [[ -z "${manifest}" ]]; then
    cat > "${out_dir}/report.md" <<EOF
# ${domain} Target Metric Eval

status: skipped

reason: pair manifest not set

Set PAIR_MANIFEST_YOUTUBE or PAIR_MANIFEST_DAVIS and rerun this script.
EOF
    echo "[target-metricpy] skip ${domain}: pair manifest not set"
    return 0
  fi
  if [[ ! -f "${manifest}" ]]; then
    cat > "${out_dir}/report.md" <<EOF
# ${domain} Target Metric Eval

status: skipped

reason: manifest missing

manifest: \`${manifest}\`
EOF
    echo "[target-metricpy] skip ${domain}: manifest missing ${manifest}"
    return 0
  fi
  python tools/run_inpainting_metric_eval.py \
    --pair_manifest "${manifest}" \
    --output_dir "${out_dir}" \
    --max_frames "${MAX_FRAMES}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    "${flag_args[@]}"
  cat > "${out_dir}/report.md" <<EOF
# ${domain} Target Metric Eval

status: complete

metric_backend: \`inference/metrics.py\`

summary: \`${out_dir}/metrics/summary.csv\`
EOF
}

write_inventory
run_domain "youtubevos" "${PAIR_MANIFEST_YOUTUBE}" "${YOUTUBE_OUT}"
run_domain "davis" "${PAIR_MANIFEST_DAVIS}" "${DAVIS_OUT}"

echo "[target-metricpy] report=${REPORT}"
echo "[target-metricpy] youtubevos=${YOUTUBE_OUT}"
echo "[target-metricpy] davis=${DAVIS_OUT}"
