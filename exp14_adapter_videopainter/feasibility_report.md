# VideoPainter Adapter Feasibility Summary

VideoPainter is a feasible future adapter target, but only after a dedicated
adapter training script is implemented in this experiment folder.

Classification:

```text
adapter_type = direct_diff_dpo_design_feasible_not_implemented
```

Why feasible:

- The upstream training loop is diffusion / DiT based.
- It samples `timesteps`, `noise`, and `noisy_video_latents`.
- It predicts velocity-like `model_pred` and compares to latent `target`.
- It already uses a mask-weighted inpainting loss.
- Region-local mask / boundary / outside weighting can be implemented over the
  latent mask resolution.

Why blocked:

- Current VideoPainter training code only handles one policy model and one clean
  reconstruction target.
- It does not load winner/loser pairs.
- It does not run frozen reference forward.
- It does not compute `m_w`, `m_l`, `m_w_ref`, `m_l_ref`.
- It does not record DPO diagnostics.

