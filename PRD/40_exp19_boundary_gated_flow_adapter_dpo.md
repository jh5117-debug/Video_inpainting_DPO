# PRD 40: Exp19 Boundary-Gated Flow-Adapter DPO

Date: 2026-06-18

## Motivation

Current best:

```text
Exp11 outer b0.75 S2
```

Exp11 improves mask-region quality and outer-boundary seams, but its DAVIS50 TC
is essentially unchanged from SFT-48000. Exp18 showed that forcing final outputs
to match warped propagated pixels is too rigid: non-oracle and oracle variants
both failed to beat Exp11. Exp19 therefore changes the role of flow from a hard
target into a light conditioning signal.

## Method

Exp19 adds zero-initialized residual flow adapters to DiffuEraser Stage2 while
keeping the Exp11 DPO objective unchanged.

Policy:

```text
Exp11 Stage2 base + flow adapter enabled
```

Reference:

```text
same Exp11 Stage2 base + flow adapter disabled
```

Base model parameters are frozen. Only the flow encoder, zero-conv residual
adapters, and alpha scales are trainable in the first gate.

## Flow Source

The intended flow source is ProPainter completed bidirectional flow from masked
inputs, without GT-error confidence and without using GT frames to estimate
confidence. Confidence is forward-backward consistency:

```text
C_flow = exp(-||F_f + Warp(F_b, F_f)||_1 / tau_flow) * valid_warp * source_valid
```

## Variants

| Variant | Gate | Extra loss | Purpose |
|---|---|---|---|
| Exp19a | `C_flow` | none | Test whether flow conditioning alone helps. |
| Exp19b | `C_flow * clip(mask + 0.75 * B_outer, 0, 1)` | none | Main boundary-gated candidate. |
| Exp19c | same as Exp19b | `0.05 L_warp` if safe | Diagnostic for latent temporal consistency. |

## Guardrails

- Do not modify Exp11 or shared training code.
- Do not train a second large flow-completion branch.
- Do not use GT to compute flow confidence.
- Do not use flow-warped pixels as final RGB targets.
- Do not run full 2000-step training until the limit=100 / 500-step gates pass.
- Do not use VBench.

## Current Status

```text
TRAINING_GATE_COMPLETED_EVAL_BLOCKED
```

The original architecture block has been recovered with an Exp19-only
hook-based wrapper:

```text
exp19_boundary_gated_flow_adapter_dpo/code/unet_motion_flow_adapter_wrapper.py
```

The wrapper injects directly at Stage2 motion-module outputs and does not use
the unsafe `additional_residuals` interfaces.

PAI gate result:

- limit100 ProPainter completed-flow cache: completed.
- zero-init / gradient preflight: passed.
- injected modules:
  - `mid_block.motion_modules.0`
  - `up_blocks.0.motion_modules.0`
  - `up_blocks.1.motion_modules.0`
- zero-init equality: passed (`mean_abs_diff = 0.0`).
- base model frozen: passed (`base_grad_norm = 0.0`).
- adapter gradient: non-zero.
- Exp19b Stage2 adapter-only 500 steps: completed.
- checkpoints: `checkpoint-250`, `checkpoint-500`, `last_weights`.

Current blocker:

```text
DAVIS10_EVAL_BLOCKED_PENDING_EXP19_INFERENCE_WRAPPER
```

The existing DAVIS evaluator can load standard DiffuEraser weights but cannot
load an external flow adapter or pass per-window flow tensors into
`pipeline_diffueraser.py`. Do not evaluate Exp19 by silently falling back to
Exp11 weights. The next required implementation is an Exp19 inference wrapper
that aligns completed-flow slices with DiffuEraser context windows.

Reports:

- `reports/exp19_isolated_wrapper_recovery_audit.md`
- `reports/exp19_isolated_wrapper_preflight.md`
- `reports/exp19b_dpo_adapter_diag_summary.md`
- `reports/exp19_final_report.md`
