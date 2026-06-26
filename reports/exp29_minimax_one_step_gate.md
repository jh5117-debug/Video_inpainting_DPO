# Exp29 MiniMax One-Step Gate

Date: 2026-06-26

Status: `MINIMAX_ONE_STEP_STRICT_RELOAD_PASSED`

The first attempt used AdamW directly on fp16 full-model parameters and became
NaN after the first update. The recorded fix used a conservative SGD micro
update (`lr=1e-7`) to verify finite update mechanics without treating this as a
final training recipe.

One-step diagnostics:

- DPO loss: 0.6931471825
- grad norm preclip: 0.8897291490
- grad max abs: 0.0529785156
- gradient tensors: 461
- strict reload missing keys: 0
- strict reload unexpected keys: 0
- one-step parameter delta probe: 2.0694979922992497e-11
- reference delta probe: 0.0

Checkpoint:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/minimax_adapter_gates_20260626/checkpoints/checkpoint-1`

