#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"

EXP_DIR="exp16_prior_confidence_gated_dpo"
MANIFEST="${EXP16_MANIFEST:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl}"
PRIOR_MANIFEST="${EXP16_PRIOR_MANIFEST:-$EXP_DIR/cache/exp16_propainter_prior_cache/manifests/exp16_train_with_prior.jsonl}"
SFT48000="${SFT48000:-/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000}"
DAVIS="${DAVIS:-/mnt/workspace/hj/nas_hj/data/external/davis_432_240}"
PROPAINTER="${PROPAINTER_WEIGHT_ROOT:-$ROOT/weights/propainter}"

mkdir -p "$EXP_DIR/reports" "$EXP_DIR/cache" "$EXP_DIR/dpo_diag" logs/pipelines reports

{
  echo "# Exp16 PAI Precheck"
  echo
  echo "- repo: $ROOT"
  echo "- source manifest: $MANIFEST"
  echo "- prior manifest: $PRIOR_MANIFEST"
  echo "- SFT-48000: $SFT48000"
  echo "- DAVIS: $DAVIS"
  echo "- ProPainter weights: $PROPAINTER"
  echo "- CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
  echo
  for path in "$MANIFEST" "$SFT48000" "$DAVIS" "$PROPAINTER"; do
    if [[ -e "$path" ]]; then
      echo "- exists: $path"
    else
      echo "- missing: $path"
    fi
  done
} > reports/exp16_prior_confidence_pai_precheck.md

python -m py_compile "$EXP_DIR"/code/*.py

if [[ ! -f "$PRIOR_MANIFEST" ]]; then
  cat > "$EXP_DIR/reports/status_runtime_blocked.md" <<EOF
# Exp16 Runtime Blocked

Exp16 did not start training.

reason: missing real ProPainter prior manifest

expected prior manifest:

\`\`\`text
$PRIOR_MANIFEST
\`\`\`

Build it first with:

\`\`\`bash
python $EXP_DIR/code/precompute_propainter_prior_cache.py \\
  --input_manifest "$MANIFEST" \\
  --output_root "$EXP_DIR/cache/exp16_propainter_prior_cache" \\
  --propainter_model_dir "$PROPAINTER" \\
  --limit 100 --resume
\`\`\`

The original generated-loser manifest must not be used directly for Exp16
training because it lacks a real ProPainter prior target.
EOF
  echo "BLOCKED: missing prior manifest $PRIOR_MANIFEST"
  exit 2
fi

python "$EXP_DIR/code/preflight_exp16.py" \
  --manifest "$PRIOR_MANIFEST" \
  --report_json "$EXP_DIR/reports/preflight_report.json" \
  --report_md "reports/exp16_preflight_report.md"

if [[ "${EXP16_ENABLE_REAL_PRIOR_X0_TRAINING:-0}" != "1" ]]; then
  echo "BLOCKED: preflight can run, but full trainer integration is intentionally disabled."
  echo "Set EXP16_ENABLE_REAL_PRIOR_X0_TRAINING=1 only after x0 prior-loss audit passes."
  exit 2
fi

echo "Exp16 training would start here only after the real prior/x0 trainer audit passes."
exit 2
