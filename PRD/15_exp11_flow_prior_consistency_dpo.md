# Exp11: Flow-Prior Consistency DPO

Experiment name:

```text
exp11_flow_prior_consistency_dpo_s1s2_2000_davis_pai
```

Goal: inherit Exp10 and add flow / ProPainter-prior / boundary consistency to
address purple haze, pasted patches, boundary discontinuity, and temporal
flicker.

Planned total loss:

```text
L_total =
    L_region_normalized_DPO
  + winner_abs_reg_weight * m_w
  + winner_gap_reg_weight * ReLU(norm_win_gap - margin)
  + lambda_flow * L_flow
  + lambda_prior * L_prior
  + lambda_boundary * L_boundary
```

First-version weights:

```text
LAMBDA_FLOW=0.1
LAMBDA_PRIOR=0.1
LAMBDA_BOUNDARY=0.2
```

Current audit result:

```text
status: blocked
```

Reason: the current Stage1/Stage2 training loops do not safely expose a
differentiable flow tensor or image/x0-space ProPainter-prior consistency
target. DAVIS inference uses ProPainter prior, but that is not the same as a
train-time prior consistency loss.

Rule: do not launch Exp11 training until
`reports/exp11_flow_prior_implementation_audit.md` passes.
