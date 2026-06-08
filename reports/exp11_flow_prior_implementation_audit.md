# Exp11 Flow / Prior Consistency Implementation Audit

Date: 2026-06-08

status: blocked

Exp11 requires train-time `L_flow`, `L_prior`, and `L_boundary` on top of Exp10 region-local normalized DPO.

Current safe implementation status:

- Region-local normalized DPO: implemented.
- Boundary-region weighted MSE diagnostics: implemented.
- DAVIS inference with ProPainter prior: implemented.
- Train-time differentiable RAFT/flow tensor: not available in Stage1/Stage2 loops.
- Train-time ProPainter prior target aligned to model output/x0: not available.
- Flow confidence statistics: not available.

Decision: do not launch Exp11 training until the missing train-time flow/prior pieces are implemented and re-audited.
