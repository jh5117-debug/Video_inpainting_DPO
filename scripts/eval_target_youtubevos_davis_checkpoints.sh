#!/usr/bin/env bash
set -euo pipefail

# Target-domain eval gate for existing checkpoints.
# This script is intentionally conservative: if the current repo cannot
# guarantee the DiffuEraser reproduction eval settings, it writes a preflight
# report and exits instead of running a mismatched evaluation.

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/logs/target_eval}"
REPORT_PATH="${REPORT_PATH:-${PROJECT_ROOT}/reports/target_domain_youtubevos_davis_eval_report.md}"
RUN_EVAL="${RUN_EVAL:-false}"

YOUTUBE_VOS_ROOT="${YOUTUBE_VOS_ROOT:-${PROJECT_ROOT}/data/external/youtubevos_432_240}"
DAVIS_ROOT="${DAVIS_ROOT:-${PROJECT_ROOT}/data/external/davis_432_240}"

BASE_WEIGHTS="${BASE_WEIGHTS:-${PROJECT_ROOT}/weights/diffuEraser/converted_weights_step48000}"
EXP3_WEIGHTS="${EXP3_WEIGHTS:-}"
NEW_EXP5_WEIGHTS="${NEW_EXP5_WEIGHTS:-}"
NEW_EXP6_WEIGHTS="${NEW_EXP6_WEIGHTS:-}"
EXP7_STAGE1_WEIGHTS="${EXP7_STAGE1_WEIGHTS:-${PROJECT_ROOT}/experiments/dpo/stage1/20260601_065618_exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500_stage1/last_weights}"
EXP7_STAGE2_WEIGHTS="${EXP7_STAGE2_WEIGHTS:-${PROJECT_ROOT}/experiments/dpo/stage2/20260601_065618_exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500_stage2/last_weights}"
HYBRID_WEIGHTS="${HYBRID_WEIGHTS:-}"

mkdir -p "${OUTPUT_ROOT}" "$(dirname "${REPORT_PATH}")"

ts="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${OUTPUT_ROOT}/preflight_${ts}"
mkdir -p "${RUN_ROOT}"

exists_status() {
  local p="$1"
  if [[ -n "${p}" && -e "${p}" ]]; then
    printf 'FOUND'
  elif [[ -n "${p}" ]]; then
    printf 'MISSING'
  else
    printf 'UNSET'
  fi
}

append_report() {
  printf '%s\n' "$*" >> "${REPORT_PATH}"
}

write_preflight_report() {
  : > "${REPORT_PATH}"
  append_report "# Target-Domain YouTube-VOS / DAVIS Eval Report"
  append_report ""
  append_report "- generated_at: \`$(date '+%F %T %Z')\`"
  append_report "- project_root: \`${PROJECT_ROOT}\`"
  append_report "- run_root: \`${RUN_ROOT}\`"
  append_report "- run_eval: \`${RUN_EVAL}\`"
  append_report ""
  append_report "## Domain Boundary"
  append_report ""
  append_report "- VideoDPO is bridge / engineering migration / ablation domain."
  append_report "- YouTube-VOS and DAVIS are the final target domains."
  append_report "- Do not treat VideoDPO partial-mask eval as final quality."
  append_report ""
  append_report "## Required Eval Settings"
  append_report ""
  append_report "| setting | required | status |"
  append_report "| --- | --- | --- |"
  append_report "| denoise steps | 6 | pending backend confirmation |"
  append_report "| PCM | disabled | pending backend confirmation |"
  append_report "| Gaussian blur | disabled | pending backend confirmation |"
  append_report "| mask dilation | none unless reproduction setting requires it | pending backend confirmation |"
  append_report "| outside mask | hard comp to winner/GT | pending backend confirmation |"
  append_report "| metrics | frame-wise metric path | pending backend confirmation |"
  append_report ""
  append_report "## Data Roots"
  append_report ""
  append_report "| dataset | status | path |"
  append_report "| --- | --- | --- |"
  append_report "| YouTube-VOS | $(exists_status "${YOUTUBE_VOS_ROOT}") | \`${YOUTUBE_VOS_ROOT}\` |"
  append_report "| DAVIS | $(exists_status "${DAVIS_ROOT}") | \`${DAVIS_ROOT}\` |"
  append_report ""
  append_report "## Checkpoints"
  append_report ""
  append_report "| label | status | path |"
  append_report "| --- | --- | --- |"
  append_report "| DiffuEraser-base/current SFT | $(exists_status "${BASE_WEIGHTS}") | \`${BASE_WEIGHTS}\` |"
  append_report "| Exp3 official_videodpo_diffueraser | $(exists_status "${EXP3_WEIGHTS}") | \`${EXP3_WEIGHTS}\` |"
  append_report "| new Exp5 winner-anchored DPO | $(exists_status "${NEW_EXP5_WEIGHTS}") | \`${NEW_EXP5_WEIGHTS}\` |"
  append_report "| new Exp6 winner-anchored DPO | $(exists_status "${NEW_EXP6_WEIGHTS}") | \`${NEW_EXP6_WEIGHTS}\` |"
  append_report "| Exp7 DPO Stage1 | $(exists_status "${EXP7_STAGE1_WEIGHTS}") | \`${EXP7_STAGE1_WEIGHTS}\` |"
  append_report "| Exp7 DPO Stage2 | $(exists_status "${EXP7_STAGE2_WEIGHTS}") | \`${EXP7_STAGE2_WEIGHTS}\` |"
  append_report "| DPO-S1 + SFT-S2 hybrid | $(exists_status "${HYBRID_WEIGHTS}") | \`${HYBRID_WEIGHTS}\` |"
  append_report ""
  append_report "## Verdict"
  append_report ""
  if [[ "${RUN_EVAL}" != "true" ]]; then
    append_report "- status: preflight only."
    append_report "- reason: \`RUN_EVAL=true\` was not set."
  else
    append_report "- status: blocked before eval."
    append_report "- reason: no target-domain eval backend has been confirmed to reproduce the required DiffuEraser settings."
  fi
  append_report ""
  append_report "## Required Questions"
  append_report ""
  append_report "- YouTube-VOS best checkpoint: pending."
  append_report "- DAVIS best checkpoint: pending."
  append_report "- Does VideoDPO-bridge DPO transfer: pending."
  append_report "- Does DPO-S1+SFT-S2 beat DPO-S1+DPO-S2: pending."
  append_report "- Should Exp9 target-domain DPO start: pending target-domain eval."
}

write_preflight_report

echo "[target-eval] report=${REPORT_PATH}"
echo "[target-eval] run_root=${RUN_ROOT}"
if [[ "${RUN_EVAL}" == "true" ]]; then
  echo "[target-eval][blocked] no confirmed target eval backend; report written instead of running mismatched eval" >&2
  exit 3
fi
