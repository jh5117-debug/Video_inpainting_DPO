# Exp29 MiniMax Trainable Forward Audit

Date: 2026-06-26

Status: `MINIMAX_TRAINABLE_FORWARD_PASSED`

MiniMax was audited as a possible true DPO adapter by constructing a native
flow-matching training forward from official code and paper evidence.

Flow target:

`z_t = t * epsilon + (1 - t) * z_0`

`target velocity = epsilon - z_0`

The audit used one fixed `davis_bear` row. The VAE was frozen. The transformer
was trainable. No optimizer step was run in this milestone.

## Results

- Original frames: 9
- Model frames: 9
- `z0` shape: `[1, 16, 3, 64, 64]`
- Transformer input shape: `[1, 48, 3, 64, 64]`
- `t`: 0.3701171875
- timestep: 370.0
- no-grad loss: 0.0171425510
- grad loss: 0.0171425510
- grad norm: 0.7473063172
- gradient tensors: 461
- max grad abs: 0.0222778320
- trainable transformer parameters: 1,127,055,424
- frozen VAE parameters: 126,892,531
- strict state_dict roundtrip missing keys: 0
- strict state_dict roundtrip unexpected keys: 0
- peak VRAM: 12561.50 MiB

## Interpretation

MiniMax is not merely an inference wrapper. The local official transformer can
participate in a native differentiable flow-matching forward with finite loss,
finite gradients, and clean key identity. This makes MiniMax a credible third
true-adapter candidate pending zero-gap, one-step strict reload, and 10-step
micro gates.

