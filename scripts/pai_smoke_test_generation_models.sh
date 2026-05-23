#!/usr/bin/env bash
set -uo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root" || exit 1

export LINGBOT_PROCESS_NAME="${LINGBOT_PROCESS_NAME:-lingbot-phy}"

stamp="$(date +%Y%m%d_%H%M%S)"
out_root="${OUT_ROOT:-$repo_root/outputs/asset_smoke_tests/$stamp}"
report="${REPORT:-$repo_root/PRD/pai_asset_smoke_report_${stamp}.md}"
run_one_sample="${RUN_ONE_SAMPLE:-0}"

mkdir -p "$out_root" "$(dirname "$report")"

if [ -f configs/paths/pai.detected.env ]; then
  # shellcheck disable=SC1091
  source configs/paths/pai.detected.env
fi

{
  echo "# PAI Asset Smoke Report"
  echo
  echo "- generated_at: $(date -Is)"
  echo "- repo_root: $repo_root"
  echo "- out_root: $out_root"
  echo "- run_one_sample: $run_one_sample"
  echo
  echo "| Model | Inference Script | Weight Root | Python Compile | Weight Listing | One-sample Generation |"
  echo "| --- | --- | --- | --- | --- | --- |"
} > "$report"

check_model() {
  local model="$1"
  local script="$2"
  local weight_var="$3"
  local weight_root="${!weight_var:-}"
  local compile_status="MISSING_SCRIPT"
  local weight_status="MISSING_WEIGHT"
  local gen_status="NOT_RUN"

  if [ -f "$script" ]; then
    python -m py_compile "$script" >/tmp/pai_smoke_compile.log 2>&1 \
      && compile_status="OK" \
      || compile_status="FAIL:$(tail -1 /tmp/pai_smoke_compile.log)"
  fi

  if [ -n "$weight_root" ] && [ -e "$weight_root" ]; then
    weight_status="OK"
    find "$weight_root" -maxdepth 2 -type f | head -20 > "$out_root/${model}_weights.txt" 2>/dev/null || true
  fi

  if [ "$run_one_sample" = "1" ]; then
    gen_status="DELEGATED_TO_CANONICAL_SMOKE_TOOL"
  fi

  printf '| %s | `%s` | `%s` | %s | %s | %s |\n' \
    "$model" "$script" "${weight_root:-UNCONFIRMED}" "$compile_status" "$weight_status" "$gen_status" >> "$report"
}

check_model diffueraser DPO_finetune/infer_diffueraser_candidate.py DIFFUERASER_WEIGHT_ROOT
check_model propainter DPO_finetune/infer_propainter_candidate.py PROPAINTER_WEIGHT_ROOT
check_model cococo DPO_finetune/infer_cococo_candidate.py COCOCO_WEIGHT_ROOT
check_model minimax_remover DPO_finetune/infer_minimax_candidate.py MINIMAX_REMOVER_WEIGHT_ROOT

cat >> "$report" <<'MD'

## Notes

- Default mode does import/compile and weight-path smoke only.
- Set `RUN_ONE_SAMPLE=1` to run canonical VideoDPO one-sample full/partial generation smoke through `tools/pai_videodpo_single_sample_generation_smoke.py`.
- This script intentionally does not start DPO training.
MD

if [ "$run_one_sample" = "1" ]; then
  {
    echo
    echo "## Canonical One-Sample Generation Smoke"
    echo
  } >> "$report"
  models_arg="${SMOKE_MODELS:-all}"
  python tools/pai_videodpo_single_sample_generation_smoke.py \
    --models "$models_arg" \
    --mask_modes "${SMOKE_MASK_MODES:-full,partial}" \
    --output_root "$out_root/canonical_videodpo_single_sample" \
    --report_path "$out_root/canonical_videodpo_single_sample/report.md" \
    --manifest_path "$out_root/canonical_videodpo_single_sample/smoke_manifest.jsonl" \
    --run_generation \
    >> "$report" 2>&1 || true
fi

echo "[smoke] report=$report"
echo "[smoke] out_root=$out_root"
