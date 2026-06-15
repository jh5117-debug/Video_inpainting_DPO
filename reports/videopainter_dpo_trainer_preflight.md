# VideoPainter DPO Trainer Preflight

Date: 2026-06-15

## Status

Trainer implemented locally; PAI preflight not run yet.

## Implemented Trainer

```text
exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py
```

The file passes local `python -m py_compile`.

It implements an isolated VideoPainter branch-adapter DPO trainer without
modifying upstream VideoPainter or shared `training/dpo` code.

## Preflight Requirement

The requested preflight requires:

- load policy VideoPainter;
- load frozen reference VideoPainter;
- load one winner / loser / mask pair;
- compute `m_w`, `m_l`, `m_w_ref`, `m_l_ref`;
- compute normalized-gap DPO loss;
- run one backward pass;
- verify reference has no gradients.

The trainer now supports this through:

```text
--preflight_only
```

The gate2000 launcher has been updated to run preflight first and only launch
2000-step training if preflight writes both:

```text
exp14_adapter_videopainter/runs/preflight/preflight_report.json
exp14_adapter_videopainter/dpo_diag/preflight_dpo_diagnostics.csv
```

## What the Trainer Defines

- `VideoPainterPairDataset` for the current frame-directory DPO manifest.
- Policy branch and frozen reference branch from the same VideoPainter branch
  checkpoint.
- Winner / loser forward passes on the same sampled noise and timestep.
- `m_w`, `m_l`, `m_w_ref`, `m_l_ref` as region-local denoising MSE.
- Exp11 outer b0.75 S2 style normalized-gap DPO:

```text
g_w = log((m_w + eps) / (m_w_ref + eps))
g_l = log((m_l + eps) / (m_l_ref + eps))
g_l_clip = clip(g_l, max=1.0)

L_DPO = mean[-logsigmoid(-0.5 * 10 * (g_w - 0.25 * g_l_clip))]
L_total = L_DPO + 0.05 * m_w + ReLU(g_w)
```

Region setting:

```text
boundary_mode = outer
mask_weight = 1.0
boundary_weight = 0.75
outside_weight = 0.05
```

## Pending PAI Check

The preflight still needs to run on PAI with real VideoPainter weights:

- `VIDEO_PAINTER_BASE_MODEL`
- `VIDEO_PAINTER_CHECKPOINT_ROOT`
- `VIDEO_PAINTER_REFERENCE_CHECKPOINT_ROOT`

If those paths are missing, or if the policy/reference forward cannot compute
finite losses, gate2000 remains blocked.

## Decision

Do not start `exp14_adapter_videopainter_gate2000` until PAI preflight passes.
