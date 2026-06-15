# Exp14: VideoPainter Adapter Feasibility

Status: feasibility / smoke planning only. No 2000-step training has been launched.

Goal:

```text
Check whether the current best DiffuEraser DPO objective
Exp11 outer b0.75 S2
can be adapted to VideoPainter.
```

Current decision:

- VideoPainter repo exists locally and has official training code.
- VideoPainter is diffusion / DiT based through CogVideoX.
- Direct Diff-DPO is structurally feasible, because the training loop exposes
  timestep, noise, target, mask, policy prediction, and inpainting loss tensors.
- The adapter is **not implemented yet**, so 1-step / 20-step smoke has not been
  run from HAL.
- Gate2000 is **not ready** until adapter code, smoke1, and smoke20 pass on PAI.

Key reports:

```text
exp14_adapter_videopainter/reports/videopainter_repo_audit.md
exp14_adapter_videopainter/reports/videopainter_training_entry_audit.md
exp14_adapter_videopainter/reports/videopainter_loss_interface_audit.md
exp14_adapter_videopainter/reports/videopainter_reference_model_audit.md
exp14_adapter_videopainter/reports/smoke1_report.md
exp14_adapter_videopainter/reports/smoke20_report.md
```

