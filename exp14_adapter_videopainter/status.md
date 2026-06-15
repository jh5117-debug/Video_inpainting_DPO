# Exp14 Status

Status: **gate2000 precheck blocked**.

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
- No implemented policy/reference DPO loss entry exists under
  `exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py`.
- PAI precheck found data and idle GPUs, but did not find Exp14 adapter code.

Decision:

```text
Do not launch 2000-step training from the upstream VideoPainter script alone.
Implement/copy a dedicated VideoPainter DPO adapter trainer first, then rerun
the gate2000 precheck. Smoke is skipped by user request, but the trainer and
reference/loss interface are still hard requirements.
```
