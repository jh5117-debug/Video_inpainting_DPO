# Exp7 current D2 partial-mask task gate

- experiment_id: `exp07_current`
- short_name: `partialmask_task_current_failed`
- status: `failed_or_suspicious_needs_prior_mask_audit`
- train_task: `true partial-mask inpainting`
- source_domain: `VideoDPO clips mostly static`
- target_domain: `partial-mask inpainting task alignment`

## What This Experiment Tests

Task changed but quality is unstable; base also poor, so eval/prior/domain/mask must be audited before more DPO.

## Registry Rule

This folder stores pointers and summaries only. Do not copy checkpoints, videos, datasets, or weights here.
