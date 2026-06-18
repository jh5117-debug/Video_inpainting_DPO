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

Implemented and run on PAI:

```text
exp18_multiframe_propagation_gated_dpo/
experiment_registry/exp18_multiframe_propagation_gated_dpo/
reports/exp18_context_and_code_audit.md
reports/exp18_propagation_confidence_audit.md
reports/exp18_x0_latent_loss_implementation_audit.md
reports/exp18_final_pai_gate_report.md
```

Execution status:

```text
PAI_GATE_COMPLETED_NEGATIVE_ABLATION
```

PAI run:

```text
worktree = /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp18_gate
cache = /mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp18_multiframe_propagation_cache_limit100
eval = /mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp18_multiframe_propagation_gated_dpo_davis10
```

Completed:

- limit=100 multi-frame propagation cache
- Exp18a Stage1-500
- Exp18b Stage1-500
- Exp18c oracle Stage1-500 diagnostic
- DAVIS10 metric sanity
- DAVIS10 visual case judgement
- dpo_diag summaries

## PAI Gate Result

DAVIS10 metric summary:

| Method | PSNR | SSIM | strict mask PSNR | boundary PSNR |
|---|---:|---:|---:|---:|
| Exp11 boundary outer b0.75 S2 | 30.2413 | 0.9650 | 18.7114 | 24.8326 |
| Exp18a prop-only S1-500 | 30.1024 | 0.9650 | 18.5725 | 24.7090 |
| Exp18b prop+gen S1-500 | 29.6892 | 0.9609 | 18.1593 | 24.7152 |
| Exp18c oracle S1-500 | 29.7626 | 0.9632 | 18.2326 | 24.7991 |
| SFT-48000 baseline | 30.0126 | 0.9635 | 18.4827 | 24.4772 |

Result:

```text
No Exp18 variant beats Exp11 outer b0.75 S2 on DAVIS10 primary metrics.
```

Visual judgement:

```text
No clearly positive Exp18-over-Exp11 case was observed.
Exp18a is the best Exp18 variant but only near-ties Exp11 in some cases.
Exp18b and Exp18c often soften details or introduce local artifacts.
```

Diagnostic judgement:

- non-oracle propagation is real but sparse;
- Exp18a/Exp18b keep high loser-dominant diagnostics;
- Exp18c oracle has high coverage but still does not beat Exp11.

Decision:

```text
Do not run Exp18 Stage1 1000, full cache, Stage1 2000, or Stage2 under the current formulation.
Keep Exp18 as an exploratory / negative ablation.
Current best remains Exp11 boundary outer b0.75 S2.
```
