#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"

mkdir -p reports

cat > reports/exp11_flow_prior_implementation_audit.md <<'EOF'
# Exp11 Flow / Prior Consistency Implementation Audit

Date: 2026-06-11

status: proxy_complete_real_flow_prior_blocked

Exp11 must not be launched from this wrapper.

Truth-audit result:

- The isolated proxy code under `exp11_flow_prior_consistency_dpo/code/` does
  not implement a real train-time ProPainter-prior loss.
- Its `L_prior` target is the frozen SFT/ref epsilon prediction, not a
  ProPainter prior frame/tensor aligned to predicted clean output.
- Its `L_flow` is an adjacent-frame residual proxy, not optical-flow or
  flow-confidence weighted warp consistency.
- Therefore existing Exp11 outputs are valid only under the label
  `Exp11-proxy: frozen-ref prior + boundary + temporal residual proxy DPO`.
  They must not be used as real flow-prior method results.

Required before Exp11 can run:

- train-time prior tensors/frames or a safe predicted-clean/x0 consistency path;
- real flow tensors or a differentiable warp target if claiming `L_flow`;
- nonzero diagnostics for prior/boundary/flow losses from the real targets;
- a new audit that marks the implementation as passed.
EOF

echo "[Exp11] proxy_complete_real_flow_prior_blocked"
echo "[Exp11] Wrote reports/exp11_flow_prior_implementation_audit.md"
echo "[Exp11] Do not start new Exp11 training from this wrapper."
exit 1
