# Exp19 Isolated Wrapper Preflight

status: `PASS`

- zero_init_mean_abs_diff: `0.0`
- preflight_loss: `0.6990330815315247`
- adapter_grad_norm: `0.0002620436257093343`
- base_grad_norm: `0.0`
- gate_stats: `{"gate_mean": 0.005275590345263481, "gate_p10": 0.0, "gate_p50": 0.0, "gate_p90": 0.0, "nonzero_gate_ratio": 0.01897321455180645}`
- alpha_values: `{"mid_block__motion_modules__0": 1.0, "up_blocks__0__motion_modules__0": 1.0, "up_blocks__1__motion_modules__0": 1.0}`
- hook_shapes: `[{"name": "mid_block.motion_modules.0", "channels": 1280, "height": 4, "width": 7, "output_type": "Tensor"}, {"name": "up_blocks.0.motion_modules.0", "channels": 1280, "height": 4, "width": 7, "output_type": "Tensor"}, {"name": "up_blocks.1.motion_modules.0", "channels": 1280, "height": 8, "width": 14, "output_type": "Tensor"}]`
