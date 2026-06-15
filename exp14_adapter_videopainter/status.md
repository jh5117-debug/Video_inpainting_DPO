# Exp14 Status

Status: **blocked before smoke**.

What passed:

- Local VideoPainter repo found.
- Official training entrypoints found.
- VideoPainter is CogVideoX / DiT diffusion-based.
- Training loop exposes timestep, noise, target, mask, model prediction, and
  checkpoint saving.
- A frozen reference model is conceptually possible by loading a second copy of
  the same pretrained branch / transformer under `torch.no_grad()`.

What has not passed:

- No VideoPainter DPO adapter training script exists yet.
- No PAI 1-step smoke was run from this HAL session.
- No 20-step smoke was run.
- No Gate2000 script should be launched.

Decision:

```text
Do not launch 2000-step training.
Implement/copy a dedicated VideoPainter DPO adapter trainer first, then run
PAI smoke1, then smoke20, then ask for user confirmation.
```

