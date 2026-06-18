# Exp19 Flow Encoder / Hook Shape Audit

Exp19b uses a lightweight 7-channel flow condition tensor: forward/backward normalized completed flow, confidence, hole mask, and outer boundary.

| module | output shape | gate |
|---|---|---|
| `mid_block.motion_modules.0` | `[B*T, 1280, 4, 7]` | `C_flow * clip(mask + 0.75 * B_outer, 0, 1)` |
| `up_blocks.0.motion_modules.0` | `[B*T, 1280, 4, 7]` | same |
| `up_blocks.1.motion_modules.0` | `[B*T, 1280, 8, 14]` | same |

Initial residual is exactly zero because 1x1 projector weights and bias are zero initialized and alpha is one. The preflight confirmed enabled adapter output matches frozen Exp11 output while first backward gives non-zero adapter gradients.
