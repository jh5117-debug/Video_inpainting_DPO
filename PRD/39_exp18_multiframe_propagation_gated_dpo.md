# PRD 39: Exp18 Multi-frame Propagation-Confidence Gated DPO

Date: 2026-06-17

## Motivation

Current best:

```text
Ours = Exp11 outer b0.75 S2
```

Exp11 proves that region-local normalized DPO with outer-boundary weighting is
effective for BR / video inpainting. However, it still treats the mask interior
mostly as a region to optimize, without explicitly deciding which pixels should
be propagated from other frames and which pixels should be generated.

Old Exp16 tried prior-confidence gating using GT-error confidence over a real
ProPainter prior cache. It was useful as implementation validation, but it did
not beat Exp11. The main weakness is conceptual: GT-error confidence is a
training oracle and does not behave like an actual video propagation mechanism.

Exp18 therefore changes the question:

```text
Which masked pixels can be reliably propagated from other frames?
Which masked pixels cannot be propagated and should be generated?
```

## Core Idea

For each target frame:

```text
M_prop = reliable propagated pixels inside the mask
M_gen = remaining pixels inside the mask
M_boundary = outer boundary
```

Policy:

```text
propagatable pixels -> preserve P_prop
non-propagatable pixels -> generate toward GT/context
outer boundary -> preserve seam consistency
```

## Data / Fixed Setting

Exp18 must inherit Exp11:

- win: GT clean video
- lose: generated rollout loser
- mask: partial mask from manifest
- base model: SFT-48000 DiffuEraser
- eval: DAVIS50, optional YouTubeVOS100 later
- protocol: raw6, no PCM, no mask dilation, no Gaussian blur, hard comp,
  frame-wise metric, no VBench

## Propagation Cache

First cache:

```text
limit = 100
method = farneback_multisource_agreement
source_window = 3
tau_conf = 0.5
write_oracle = true
```

Outputs:

```text
propagated_frames/
confidence_maps/
source_index_maps/
source_count_maps/
reliable_masks/
generate_masks/
oracle_confidence_maps/
manifests/exp18_train_with_multiframe_prop_limit100.jsonl
```

Non-oracle confidence:

```text
C_prop = source_valid * flow_consistency * multi_source_agreement * source_count_score
```

Oracle confidence:

```text
oracle_conf = exp(-alpha * |P_prop - GT|)
```

Oracle is diagnostic only and cannot be reported as the final method.

## Loss

Base loss is Exp11 outer b0.75 S2:

```text
L_base = region-local normalized-gap clipped-loser-gap winner-anchored DPO
```

New latent losses:

```text
z_prop = VAE(P_prop)
z_gt = VAE(GT)
z_hat_x0 = predicted clean latent from model output
```

```text
L_prop = sum(M * C_prop * |z_hat_x0 - z_prop|) / (sum(M * C_prop) + eps)
L_gen = sum(M_gen * |z_hat_x0 - z_gt|) / (sum(M_gen) + eps)
L_boundary = sum(B_outer * |z_hat_x0 - z_gt|) / (sum(B_outer) + eps)
```

Total:

```text
L_total =
    L_base
    + lambda_prop * L_prop
    + lambda_gen * L_gen
    + lambda_boundary_extra * L_boundary
```

Defaults:

```text
lambda_prop = 0.1
lambda_gen = 0.05
lambda_boundary_extra = 0.1
```

## Variants

| Variant | Confidence | Loss extras | Purpose |
|---|---|---|---|
| Exp18a | non-oracle flow/agreement | `0.1 L_prop + 0.1 L_boundary` | Test whether preserving propagated pixels helps. |
| Exp18b | non-oracle flow/agreement | `0.1 L_prop + 0.05 L_gen + 0.1 L_boundary` | Test propagation/generation split. |
| Exp18c | oracle upper bound | `0.1 L_prop + 0.05 L_gen + 0.1 L_boundary` | Diagnose whether confidence is the bottleneck. Not paper method. |

## Run Plan

Do not run full training first.

First gate:

```text
limit=100 propagation cache
cache quality audit
Exp18a Stage1 500
Exp18b Stage1 500
Exp18c Stage1 500 diagnostic
DAVIS10 visual + metric
```

Decision:

- if Exp18a/b beats Exp11 visually and numerically, extend best non-oracle to
  Stage1 1000 and DAVIS50 quick eval;
- if only Exp18c improves, confidence estimation is the bottleneck;
- if Exp18c also fails, pause Exp18.

## Current Status

Implemented on HAL:

```text
exp18_multiframe_propagation_gated_dpo/
experiment_registry/exp18_multiframe_propagation_gated_dpo/
reports/exp18_context_and_code_audit.md
reports/exp18_propagation_confidence_audit.md
reports/exp18_x0_latent_loss_implementation_audit.md
```

Current execution status:

```text
IMPLEMENTATION_READY_ON_HAL
PAI_RUN_BLOCKED_IN_THIS_SESSION_BY_MISSING_PAI_MOUNT_OR_SSH
```

No Exp18 result exists yet. Do not claim Exp18 improvement until real PAI cache,
training diagnostics, DAVIS metrics, and visual evidence are available.

