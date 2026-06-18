# Exp19 Isolated Wrapper Recovery Audit

## Previous Blocker

The earlier implementation attempted to use `down_block_additional_residuals`, `mid_block_additional_residual`, and `down_intrablock_additional_residuals`. Those are ControlNet / T2I-Adapter style contracts and were unsafe for Exp19 because they can double-add residuals or enforce down/mid shape contracts that do not match temporal multi-scale flow injection.

## Implemented Recovery

- Implemented `exp19_boundary_gated_flow_adapter_dpo/code/unet_motion_flow_adapter_wrapper.py`.
- The wrapper registers forward hooks on explicit Stage2 motion modules rather than using `additional_residuals`.
- Selected modules:
  - `mid_block.motion_modules.0`: output `[B*T, 1280, 4, 7]`
  - `up_blocks.0.motion_modules.0`: output `[B*T, 1280, 4, 7]`
  - `up_blocks.1.motion_modules.0`: output `[B*T, 1280, 8, 14]`
- Reference forward disables the adapter on the same frozen Exp11 base.

## Bug Fixed During PAI Recovery

The first hook wrapper checked the unsanitized module name but stored projectors under a sanitized key, causing projectors to be rebuilt every forward. The optimizer then tracked stale parameters and `adapter_grad_norm` stayed zero. The keying bug was fixed by checking the sanitized key consistently.

## Preflight Result

```text
# Exp19 Isolated Wrapper Preflight

status: `PASS`

- zero_init_mean_abs_diff: `0.0`
- preflight_loss: `0.6990330815315247`
- adapter_grad_norm: `0.0002620436257093343`
- base_grad_norm: `0.0`
- gate_stats: `{"gate_mean": 0.005275590345263481, "gate_p10": 0.0, "gate_p50": 0.0, "gate_p90": 0.0, "nonzero_gate_ratio": 0.01897321455180645}`
- alpha_values: `{"mid_block__motion_modules__0": 1.0, "up_blocks__0__motion_modules__0": 1.0, "up_blocks__1__motion_modules__0": 1.0}`
- hook_shapes: `[{"name": "mid_block.motion_modules.0", "channels": 1280, "height": 4, "width": 7, "output_type": "Tensor"}, {"name": "up_blocks.0.motion_modules.0", "channels": 1280, "height": 4, "width": 7, "output_type": "Tensor"}, {"name": "up_blocks.1.motion_modules.0", "channels": 1280, "height": 8, "width": 14, "output_type": "Tensor"}]`

```
