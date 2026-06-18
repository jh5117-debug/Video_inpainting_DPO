# Exp19 Inference Checkpoint Loading Audit

- checkpoint: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/exp19b_boundary_flow_adapter_s2_500_limit100/last_weights/flow_adapter.pt`
- target_modules: `['mid_block.motion_modules.0', 'up_blocks.0.motion_modules.0', 'up_blocks.1.motion_modules.0']`
- tensors: `9`
- nonzero_tensors: `9`
- parameter_count: `30723`
- parameter_l2_norm: `1.7327120735494161`
- strict_load: `true`
- missing_keys: `[]`
- unexpected_keys: `[]`
- fallback_used: `false`
- adapter_nonzero: `true`

## Keys

- `mid_block__motion_modules__0.alpha` norm=1.0 sha256=e00e5eb9444182f3
- `mid_block__motion_modules__0.proj.bias` norm=0.00984306912869215 sha256=29f5a076209ca705
- `mid_block__motion_modules__0.proj.weight` norm=0.02425878681242466 sha256=cbb5d555b6854b6b
- `up_blocks__0__motion_modules__0.alpha` norm=1.0 sha256=e00e5eb9444182f3
- `up_blocks__0__motion_modules__0.proj.bias` norm=0.009938680566847324 sha256=17746d4f7d7cc17a
- `up_blocks__0__motion_modules__0.proj.weight` norm=0.024205094203352928 sha256=60d0d679dad7d01f
- `up_blocks__1__motion_modules__0.alpha` norm=1.0 sha256=e00e5eb9444182f3
- `up_blocks__1__motion_modules__0.proj.bias` norm=0.012081841006875038 sha256=47299277e426b818
- `up_blocks__1__motion_modules__0.proj.weight` norm=0.027840981259942055 sha256=5ebc68771152e1ae
