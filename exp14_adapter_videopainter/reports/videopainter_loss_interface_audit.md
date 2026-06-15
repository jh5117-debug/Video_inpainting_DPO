# VideoPainter Loss Interface Audit

Status: direct Diff-DPO design is feasible, but not implemented.

## Required DPO Quantities

```text
m_w     = policy winner loss
m_l     = policy loser loss
m_w_ref = frozen reference winner loss
m_l_ref = frozen reference loser loss
```

## VideoPainter Compatibility

1. Diffusion / noise-prediction model: yes. It uses CogVideoX / DiT, timesteps,
   noise, noisy latents, and velocity prediction.
2. Same timestep/noise for winner and loser: feasible after adapter code changes.
3. Policy loss and reference loss: feasible after loading a frozen reference copy.
4. Frozen reference model: feasible in principle, not implemented upstream.
5. Region-local weighting: feasible because masks are interpolated to latent
   resolution in the training loop.
6. dpo_diag: not implemented upstream; must be added.

## Adapter Type

```text
adapter_type = direct_diff_dpo_design_feasible_not_implemented
```

This is stronger than output-level preference because the model exposes latent
denoising loss tensors. But it is not ready for smoke until the adapter trainer
exists.

## Proposed Exp11 Outer Loss

```text
g_w = log((m_w + eps) / (m_w_ref + eps))
g_l = log((m_l + eps) / (m_l_ref + eps))
g_l_clip = clip(g_l, max=1.0)

L_DPO = mean[-logsigmoid(-0.5 * 10 * (g_w - 0.25 * g_l_clip))]
L_total = L_DPO + 0.05 * m_w + ReLU(g_w)
```

Region settings:

```text
mask = 1.0
boundary_outer = 0.75
outside = 0.05
```

## Blocking Gaps

- No winner/loser pair dataset in VideoPainter format.
- No policy/reference dual forward.
- No DPO loss implementation.
- No adapter diagnostics.
- No PAI smoke memory measurement.

## Decision

Do not run 1-step or 20-step smoke yet. First implement the adapter trainer in
the isolated `exp14_adapter_videopainter/code/` directory.

