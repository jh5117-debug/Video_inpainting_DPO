# Exp14: VideoPainter Adapter Feasibility

Status: direct-gate precheck blocked. No 2000-step training has been launched.

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
- The adapter is **not implemented yet**.
- The user requested skipping smoke, but Gate2000 is still blocked until the
  isolated trainer and minimum preflight pass on PAI.

Key reports:

```text
exp14_adapter_videopainter/reports/videopainter_repo_audit.md
exp14_adapter_videopainter/reports/videopainter_training_entry_audit.md
exp14_adapter_videopainter/reports/videopainter_loss_interface_audit.md
exp14_adapter_videopainter/reports/videopainter_reference_model_audit.md
exp14_adapter_videopainter/reports/smoke1_report.md
exp14_adapter_videopainter/reports/smoke20_report.md
```
