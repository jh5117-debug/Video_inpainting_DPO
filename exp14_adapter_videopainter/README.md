# Exp14: VideoPainter Adapter Feasibility

Status: isolated trainer implemented; no 2000-step training has been launched.

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
- The adapter trainer now exists in this isolated experiment folder:
  `exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py`.
- The user requested skipping smoke, but Gate2000 is still blocked until the
  minimum trainer preflight passes on PAI.
- The gate launcher now runs `--preflight_only` before launching 2000-step.
  If policy/reference winner/loser loss, backward, or diagnostics fail, it
  blocks instead of falling back to upstream VideoPainter training.

Key reports:

```text
exp14_adapter_videopainter/reports/videopainter_repo_audit.md
exp14_adapter_videopainter/reports/videopainter_training_entry_audit.md
exp14_adapter_videopainter/reports/videopainter_loss_interface_audit.md
exp14_adapter_videopainter/reports/videopainter_reference_model_audit.md
exp14_adapter_videopainter/reports/smoke1_report.md
exp14_adapter_videopainter/reports/smoke20_report.md
```

Current limitation:

- The isolated trainer is a branch-adapter DPO trainer, not the upstream
  VideoPainter objective.
- It has not yet been validated on PAI with the actual VideoPainter weights.
- Multi-GPU sharding is not implemented; preflight must determine whether the
  single-process branch/reference setup is memory-feasible.
