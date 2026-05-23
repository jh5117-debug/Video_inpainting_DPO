#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root" || exit 1

export MODEL_NAME="${MODEL_NAME:-all}"
export NUM_MASKS_PER_VIDEO="${NUM_MASKS_PER_VIDEO:-4}"
bash scripts/pai_generate_partialmask_losers_k4.sh
