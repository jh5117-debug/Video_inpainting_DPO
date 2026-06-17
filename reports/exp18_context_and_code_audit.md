# Exp18 Context And Code Audit

Date: 2026-06-17

## Current Best

Current best remains:

```text
Ours = Exp11 outer b0.75 S2
```

Evidence:

- code: `exp11_region_boundary_ablation/code/train_stage1.py`
- code: `exp11_region_boundary_ablation/code/train_stage2.py`
- registry: `experiment_registry/exp11_region_boundary_ablation/`
- Stage2 last weights:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/last_weights`

Metrics:

| Dataset | Method | PSNR | SSIM | LPIPS | VFID | TC | Mask PSNR |
|---|---|---:|---:|---:|---:|---:|---:|
| DAVIS50 | Exp11 outer b0.75 S2 | 33.0140 | 0.9723 | 0.0154 | 0.1754 | 0.9711 | 24.1675 |
| YouTubeVOS100 | Exp11 outer b0.75 S2 | 33.7238 | 0.9711 | 0.0168 | 0.1925 | 0.9821 | 24.7532 |

## Fixed Setting To Inherit

Exp18 must inherit the Exp11 fixed setting:

- win: GT clean video
- lose: generated rollout loser
- mask: partial mask from manifest
- model: SFT-48000 DiffuEraser
- prior/inference setting: ProPainter as in DiffuEraser pipeline
- eval protocol: raw6, no PCM, no mask dilation, no Gaussian blur, hard comp, frame-wise metric, no VBench

## Mask Convention

Current DiffuEraser DPO dataset convention:

```text
brushnet mask: 0 = hole / unknown, 1 = known outside
Exp18 loss mask: 1 = hole
hole = 1 - brushnet_mask
```

## x0 / Latent Feasibility

Exp16 already implemented and passed a Stage1 small gate with:

```text
z_hat_x0 = predicted clean latent reconstructed from model output
z_gt = VAE(GT)
z_prior = VAE(real ProPainter prior)
```

Reusable helper:

```text
exp16_prior_confidence_gated_dpo/code/exp16_loss.py
  predict_x0_from_model_output()
```

Exp18 copies this idea but replaces `z_prior` with:

```text
z_prop = VAE(P_prop)
```

where `P_prop` must come from multi-frame propagation cache.

## Available Flow / Propagation Code

Local audit found multiple flow-related assets:

- `propainter/RAFT/`
- `propainter/model/modules/flow_comp_raft.py`
- `propainter/utils/flow_util.py`
- `weights/propainter`
- `/home/hj/Video_inpainting_DPO/weights/propainter/raft-things.pth`
- `/home/hj/.cache/vbench/raft_model/models/raft-things.pth`

For the first Exp18 implementation, the cache script uses a runnable
non-oracle OpenCV Farneback optical-flow path with:

- target-to-source backward warping;
- source-valid mask check;
- forward-backward consistency;
- multi-source RGB agreement;
- confidence-weighted propagated pixels.

This is a conservative implementation path that does not require changing
ProPainter internals. RAFT/ProPainter-flow can replace the flow estimator later.

## New Exp18 Code

Created:

```text
exp18_multiframe_propagation_gated_dpo/
  code/precompute_multiframe_propagation_cache.py
  code/exp18_dataset.py
  code/exp18_loss.py
  code/train_exp18_stage1.py
  code/exp18_dpo_diag.py
  code/eval_exp18_variants.py
  code/make_exp18_visuals.py
```

Important guardrails:

- `Exp18PropagationManifestDataset` refuses rows without propagation frames and confidence maps.
- `train_exp18_stage1.py` reads `propagation_confidence` from the cache manifest.
- It does not compute confidence from GT-error.
- It does not use ProPainter prior directly as propagation.
- It does not use frozen-reference epsilon proxy.

## PAI Accessibility In This HAL Session

This HAL session cannot run the PAI experiment directly:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO: not visible
/mnt/workspace/hj/nas_hj/data/external/davis_432_240: not visible
ssh pai: hostname cannot be resolved
```

Therefore current status is:

```text
IMPLEMENTATION_READY_ON_HAL
PAI_RUN_BLOCKED_IN_THIS_SESSION_BY_MISSING_PAI_MOUNT_OR_SSH
```

This is not an algorithmic block. It means the Exp18 scripts should be run from
the PAI Codex/session where the training manifest, data, weights, and GPUs are
visible.

