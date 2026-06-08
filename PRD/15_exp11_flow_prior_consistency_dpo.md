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

## 2026-06-08 GPU Availability Note

PAI GPU0-3 were observed as mostly free while Exp9 used GPU4-7. This does not
change the Exp11 launch rule. Hardware availability is not the blocker; the
blocker is missing safe train-time implementation for:

- differentiable flow consistency;
- model-output / predicted-clean-sample versus ProPainter-prior consistency;
- audited boundary consistency integrated with the DPO Stage1/Stage2 loops.

Until those pieces are implemented and the audit passes, Exp11 must remain a
prepared but blocked experiment. Do not start it by setting
`EXP11_ENABLE_TRAINING=1` merely because GPUs are idle.

## 2026-06-09 Frame-Length Rule

When Exp11 becomes trainable, it must inherit the Exp9/10 target-domain frame
rule:

```text
NFRAMES=24
DAVIS_VIDEO_LENGTH=24
```

Do not run 16-frame DAVIS validation. The DiffuEraser/ProPainter inference path
requires effective video, mask, and prior duration greater than 22 frames.
