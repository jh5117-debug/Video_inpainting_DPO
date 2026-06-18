# Exp19 Context and Architecture Audit

Date: 2026-06-18

## Current Best Baseline

Current best remains:

```text
Exp11 outer b0.75 S2
```

Paths from the Exp11 registry:

```text
stage1 = /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s1_2000_davis_pai/last_weights
stage2 = /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/last_weights
```

Exp11 loss:

```text
region-local normalized-gap clipped-loser-gap winner-anchored DPO
boundary_mode = outer
boundary_weight = 0.75
outside_weight = 0.05
```

## ProPainter Completed Flow

ProPainter can compute and complete bidirectional adjacent flow:

- RAFT source: `propainter/model/modules/flow_comp_raft.py`
- flow completion: `propainter/model/recurrent_flow_completion.py`
- ProPainter inference path: `propainter/inference.py`

Observed tensor contract:

```text
forward flow:  B, T-1, 2, H, W
backward flow: B, T-1, 2, H, W
```

Flow convention is pixel displacement in image coordinates. The Exp19 exporter
uses masked input frames to avoid using unmasked GT pixels inside the hole.

Implemented isolated exporter:

```text
exp19_boundary_gated_flow_adapter_dpo/code/export_propainter_completed_flow.py
```

## FloED Relationship

FloED uses:

- a dedicated flow completion branch
- multi-scale flow adapters
- latent interpolation
- flow attention cache

Exp19 does not claim to reproduce FloED. Exp19 proposes a lighter adapter over
DiffuEraser/Exp11:

- reuse ProPainter completed flow
- add zero-initialized residual adapters
- train with the Exp11 localized DPO objective
- gate by flow confidence and outer boundary

## Stage2 Injection Audit

The shared `libs/unet_motion_model.py` exposes:

```text
down_block_additional_residuals
down_intrablock_additional_residuals
mid_block_additional_residual
```

However, the current forward path is not safe for the requested multi-scale
down+mid adapter without changing shared code:

1. Passing both `down_block_additional_residuals` and
   `mid_block_additional_residual` activates a ControlNet-style branch.
2. The same `down_block_additional_residuals` are then added again by an
   unconditional second branch.
3. Passing only down residuals falls into the legacy T2I-adapter path, whose
   shape contract is not the same as the full list of UNet skip residuals.
4. A mid-block-only adapter would be possible, but it is not the requested
   multi-scale Exp19 method.

## Reference Model Design

The intended memory-saving policy/reference design is feasible in principle:

```text
policy = Exp11 base + adapter enabled
reference = same Exp11 base + adapter disabled, no_grad, eval
```

But this is only valid after the adapter injection interface is safe and after
an Exp19 inference wrapper can pass flow tensors through the denoising loop.

## Eval Wrapper Audit

The existing fixed DAVIS wrapper:

```text
tools/run_davis50_framewise_protocol_eval.py
```

loads standard DiffuEraser `last_weights`. It cannot load an external flow
encoder / adapter state dict and cannot provide per-video flow tensors to the
UNet. Therefore Exp19 also needs a custom isolated inference wrapper before any
metric can be trusted.

## Decision

```text
BLOCKED_MULTI_SCALE_INJECTION_UNSAFE
```

Do not launch Exp19 training yet.

Allowed next implementation path:

1. Copy `libs/unet_motion_model.py` into Exp19.
2. Implement a clean Exp19-only residual interface in the copied model.
3. Write an Exp19 inference wrapper that exports completed flow per DAVIS video,
   loads adapter weights, and still calls the existing metric backend.
4. Re-run preflight, then limit=100 flow cache, then 500-step gates.
