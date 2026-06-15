# Exp14 Status

Status: **isolated trainer implemented; gate2000 blocked until PAI preflight passes**.

What passed:

- Local VideoPainter repo found.
- Official training entrypoints found.
- VideoPainter is CogVideoX / DiT diffusion-based.
- Training loop exposes timestep, noise, target, mask, model prediction, and
  checkpoint saving.
- A frozen reference model is conceptually possible by loading a second copy of
  the same pretrained branch / transformer under `torch.no_grad()`.

What changed on 2026-06-15:

- Added isolated trainer:
  `exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py`
- The trainer reads the current frame-directory DPO manifest with
  `win_video_path`, `final_loser_video_path`, and `mask_path`.
- It defines policy/reference VideoPainter branch forwards on shared
  timestep/noise and computes `m_w`, `m_l`, `m_w_ref`, `m_l_ref`.
- It implements Exp11 outer b0.75 S2 style region-local normalized-gap
  clipped-loser-gap winner-anchored DPO.
- It writes `dpo_diagnostics.csv` and supports `--preflight_only`.

What has not passed yet:

- PAI has not run the required trainer preflight yet.
- The 2000-step gate has not been launched.
- Multi-GPU sharding is not implemented in the isolated trainer; the first
  PAI preflight must verify memory feasibility.

Decision:

```text
Do not launch 2000-step training from the upstream VideoPainter script alone.
Rerun the gate2000 script after syncing Exp14 to PAI; it now runs
`--preflight_only` first and launches gate2000 only if policy/reference DPO loss
and backward pass succeed.
```
