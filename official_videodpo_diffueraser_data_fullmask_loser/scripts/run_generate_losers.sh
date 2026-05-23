#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/lingbot_process.sh"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: run_generate_losers.sh [extra tools.offline_loser_generation args]

Plans full-mask offline loser generation. By default this is a dry run.
Set RUN_REAL=1 only after model env/weights/data paths are confirmed.
EOF
  exit 0
fi

MODEL_NAME="${MODEL_NAME:-diffueraser}"
SOURCE_DATASET="${SOURCE_DATASET:-${VIDEO_DPO_DATA_ROOT:-}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${GENERATED_LOSER_ROOT:-$REPO_ROOT/data/generated_losers/fullmask}}"
MANIFEST="${MANIFEST:-$OUTPUT_ROOT/manifest.schema.json}"

COMMON_ARGS=(
  -m tools.offline_loser_generation
  --source_dataset "${SOURCE_DATASET:-MISSING_VIDEO_DPO_DATA_ROOT}"
  --output_root "$OUTPUT_ROOT"
  --model_name "$MODEL_NAME"
  --mask_mode full
  --comp false
  --offline true
  --seed "${SEED:-42}"
  --save_manifest "$MANIFEST"
)

if [[ "${RUN_REAL:-0}" != "1" ]]; then
  lingbot_run_python "${COMMON_ARGS[@]}" --dry_run --allow_missing_assets "$@"
  exit 0
fi

lingbot_exec_python "${COMMON_ARGS[@]}" "$@"
