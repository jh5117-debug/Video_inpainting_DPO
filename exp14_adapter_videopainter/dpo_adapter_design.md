# VideoPainter DPO Adapter Design

This design adapts the current best DiffuEraser DPO method:

```text
Exp11 outer b0.75 S2
region-local normalized-gap clipped-loser-gap winner-anchored DPO
```

## Proposed Loss

For the same noise and timestep:

```text
m_w     = region_weighted_mse(policy(winner), winner_target)
m_l     = region_weighted_mse(policy(loser),  loser_target)
m_w_ref = region_weighted_mse(reference(winner), winner_target)
m_l_ref = region_weighted_mse(reference(loser),  loser_target)

g_w = log((m_w + eps) / (m_w_ref + eps))
g_l = log((m_l + eps) / (m_l_ref + eps))
g_l_clip = clamp(g_l, max=1.0)

L_DPO = mean[-logsigmoid(-0.5 * 10 * (g_w - 0.25 * g_l_clip))]
L_total = L_DPO + 0.05 * m_w + ReLU(g_w)
```

Region weights:

```text
mask = 1.0
boundary_outer = 0.75
outside = 0.05
```

Boundary convention:

```text
mask == 1 means hole / target region
boundary_outer = dilate(mask) - mask
```

## Required Code Changes

These changes must be made only under `exp14_adapter_videopainter/code/`, not
in upstream VideoPainter and not in shared `training/dpo`.

1. Convert current D3/YouTube-VOS GT-win manifest into a VideoPainter-compatible
   pair dataset.
2. Add winner and loser batch tensors.
3. Load a trainable policy branch/transformer and a frozen reference copy.
4. Reuse identical timestep/noise for policy and reference forward.
5. Compute region-local MSE on the latent prediction target.
6. Record `adapter_diag.csv` / `dpo_diagnostics.csv`.
7. Save smoke checkpoints and verify reference parameters have no gradients.

## Smoke Requirements

Smoke1:

- 1 optimization step.
- finite loss.
- backward succeeds.
- checkpoint saves.
- reference frozen.

Smoke20:

- 20 optimization steps.
- diagnostics saved.
- tiny DAVIS validation on 2-3 videos.
- four-column visual output if inference wrapper is connected.

