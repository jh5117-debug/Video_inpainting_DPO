# Commands

No commands are executed from this registry file.

## Training / Eval Command Status

- Registry status: `partial_metrics_complete_pai_diag_pending`
- Any missing PAI commands or logs must be recovered by the PAI manual audit block.
- Do not rerun training from this file.

## Intended Configuration Snapshot

```text
train_task = True partial-mask inpainting task
train_mask_mode = partial
mask_from_manifest = true
loss_region_mode = full
beta_dpo = 10
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
lose_gap_weight = 0.25
```
