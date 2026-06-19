#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

python -m py_compile exp20_autoresearch_scale_adaptive_region_dpo/code/*.py
python -m unittest discover -s exp20_autoresearch_scale_adaptive_region_dpo/tests -p 'test_*.py'
python exp20_autoresearch_scale_adaptive_region_dpo/code/search_controller.py --init-roots

echo "Exp20 preflight complete. Heavy training remains gated by full trainer parity."
