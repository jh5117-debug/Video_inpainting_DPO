# Exp43 H20 BF16 Safe Preflight

Status: `H20_EXP43_BF16_SAFE_READY`

- Output root: `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp43_h20_minimax_stage2_sft_runner/bf16_preflight_20260629_094430`
- Torch/CUDA: `2.5.1+cu124` / `12.4`
- BF16 supported: `True`

| case | status | ranks | world size | rank0 loss | rank0 grad norm | checkpoint | rank0 peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| P0 | PASS | 1 | 1 | 2050.82861328125 | 2.831369161605835 |  | 132.0 |
| P1 | PASS | 1 | 1 |  |  |  | 6260.286 |
| P2 | PASS | 1 | 1 | 0.19614098966121674 | 0.0 | not_requested | 7274.638 |
| P3 | PASS | 1 | 1 | 0.19614098966121674 | 3.281572142587461 | not_requested | 59932.281 |
| P4 | PASS | 1 | 1 | 0.19492636620998383 | 3.179090593773888 | PASS | 68982.773 |
| P5 | PASS | 1 | 1 | 0.19614098966121674 | 3.281572142587461 | PASS | 59932.281 |
| P6 | PASS | 2 | 2 | 0.19614098966121674 | 2.1718841431594367 | PASS | 62087.76 |
| P7 | PASS | 8 | 8 | 0.19614098966121674 | 1.488086613488801 | PASS | 62087.76 |

Policy:

- VAE encode/decode fp32.
- Transformer bf16 for bf16 cases.
- Loss, residual, reduction, and gradient norm fp32.
- Flash and memory-efficient SDPA disabled when PyTorch exposes backend toggles.
- xFormers/flash-attn disabled by environment flags.
- Timesteps clamped away from exact 0/1.
- GradScaler disabled for bf16.
- No silent fallback is accepted.

This is runtime stability evidence only. It does not claim MiniMax quality improvement.
