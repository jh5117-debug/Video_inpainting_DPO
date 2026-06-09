# Commands

No commands are executed from this registry file.

## Training / Eval Command Status

- Registry status: `planned_or_blocked_until_regionloss_verified`
- Any missing PAI commands or logs must be recovered by the PAI manual audit block.
- Do not rerun training from this file.

## Intended Configuration Snapshot

```text
train_task = Target-domain partial-mask inpainting
train_mask_mode = partial
mask_from_manifest = true
loss_region_mode = region
beta_dpo = 10
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
lose_gap_weight = 0.25
```
