# Exp11: Flow-Prior Consistency DPO

## Current Status

Status: blocked.

Do not launch Exp11 by simply reusing Exp10 region-local DPO settings. That
would create a mislabeled experiment. Exp11 requires train-time flow/prior
consistency losses to be implemented and audited.

Required before launch:

- `L_flow` has a real train-time differentiable temporal warp target.
- `L_prior` compares the model output / x0 target against a valid ProPainter
  prior target in the correct space.
- `L_boundary` is wired into the training loss and diagnostics.
- `reports/exp11_flow_prior_implementation_audit.md` is updated from blocked to
  passed.

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

## 2026-06-11 Truth Audit Correction

Status: valid as `Exp11-proxy` only; blocked as real flow-prior consistency.

The 2026-06-11 truth audit found that the existing Exp11 run is a proxy
implementation, not a real optical-flow / ProPainter-prior consistency DPO
implementation:

- `L_prior` uses frozen SFT / reference epsilon prediction as a proxy target,
  not train-time ProPainter prior frames or tensors aligned to predicted clean
  output.
- `L_flow` is an adjacent-frame residual proxy, not optical-flow or
  flow-confidence weighted warp consistency.
- old Exp11 metric rows are therefore usable only under the label
  `Exp11-proxy: frozen-ref prior + boundary + temporal residual proxy DPO`.
  They must not be reported as real flow-prior consistency method results.

The canonical Exp11 launcher now writes a blocked audit report and exits before
training. Exp11 can be re-enabled only after a new implementation audit confirms
real train-time prior and, if claimed, real flow targets.

## 2026-06-11 Existing Run Audit

Status: `Exp11-proxy`, not real optical-flow / ProPainter-prior consistency DPO.

- Stage1 complete: `true`.
- Stage2 complete: `true`.
- Stage1 last weights: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s1_2000_davis_pai/last_weights`.
- Stage2 last weights: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s2_2000_davis_pai/last_weights`.
- Stage1 dpo_diag: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s1_2000_davis_pai/dpo_diagnostics.csv`.
- Stage2 dpo_diag: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s2_2000_davis_pai/dpo_diagnostics.csv`.
- Existing whole-frame / bbox all-metric DAVIS eval complete: `true`.
- Strict mask-pixel metrics present in existing eval: `false`.

Correct label:

```text
Exp11-proxy: frozen-ref prior + boundary + temporal residual proxy DPO
```

Future item:

```text
Exp11-real: image-space ProPainter prior tensor + optical-flow warp consistency + flow confidence mask
```

Do not retrain Exp11-proxy unless Stage1/Stage2 weights or dpo_diag are missing.
They are present in the audited run.

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

When Exp11 becomes trainable on the current non-regeneration D3 data, it must
inherit the Exp9/10 target-domain frame rule:

```text
NFRAMES=16
DAVIS_VIDEO_LENGTH=24
```

Do not run 16-frame DAVIS validation. The DiffuEraser/ProPainter inference path
requires effective video, mask, and prior duration greater than 22 frames. Do not
pad or repeat frames to fake 24-frame training; true 24-frame training requires
regenerating D3 clips at that length.
