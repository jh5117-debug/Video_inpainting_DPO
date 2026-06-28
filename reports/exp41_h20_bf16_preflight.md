# Exp41 H20 MiniMax BF16 Preflight

Status: `H20_MINIMAX_BF16_SAFE_READY`

H20 BF16/SIGFPE runtime preflight completed on `instance-afs92r3e` without
source-code modification. PAI was not used. The run used Exp41-only helper and
launcher files under `exp41_h20_minimax_parallel_bf16/`.

## Runtime

- H20 output root: `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp41_h20_minimax_parallel_bf16/bf16_preflight_20260629_064017`
- Torch/CUDA: `2.5.1+cu124` / `12.4`
- BF16 supported: `True`
- MiniMax weights: `/home/nvme01/H20_Video_inpainting_DPO/weights/minimax_remover/current`
- MiniMax repo: `/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax/pai_abs/mnt/nas/hj/official_repos/MiniMax-Remover_28e12b4`
- Manifest: `exp41_h20_minimax_parallel_bf16/manifests/exp41_exp40_localdpo_v3_train_h20.jsonl`

## Safe Policy Used

- VAE encode/decode ran fp32 in the Exp41 helper.
- DiT / MiniMax transformer ran bf16 for P2/P3/P5/P6/P7.
- Flow loss and residual target comparison used fp32 reduction.
- Flash and memory-efficient SDPA were disabled where PyTorch exposes the
  backend toggles; math SDPA was enabled.
- xFormers/flash-attn were disabled through environment flags.
- Timestep values were away from exact 0/1.
- No source files in MiniMax official repo, `inference/metrics.py`, shared
  trainer, or Exp1-Exp40 code were modified.

## Matrix

| case | status | world size | ranks | dtype | rank0 loss | rank0 grad norm | checkpoint | rank0 peak MiB |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: |
| P0 | PASS | 1 | 1 | bf16 | 2050.82861328125 | 2.831369161605835 |  | 132.0 |
| P1 | PASS | 1 | 1 | torch.float32 |  |  |  | 6260.286 |
| P2 | PASS | 1 | 1 | torch.bfloat16 | 0.19614098966121674 | 0.0 | not_requested | 7274.638 |
| P3 | PASS | 1 | 1 | torch.bfloat16 | 0.19614098966121674 | 3.281572142587461 | not_requested | 59932.281 |
| P4 | PASS | 1 | 1 | torch.float32 | 0.19492636620998383 | 3.179090593773888 | PASS | 68982.773 |
| P5 | PASS | 1 | 1 | torch.bfloat16 | 0.19614098966121674 | 3.281572142587461 | PASS | 59932.281 |
| P6 | PASS | 2 | 2 | torch.bfloat16 | 0.19614098966121674 | 2.1718841431594367 | PASS | 62087.76 |
| P7 | PASS | 8 | 8 | torch.bfloat16 | 0.19614098966121674 | 1.488086613488801 | PASS | 62087.76 |

## Warnings

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` is not supported on this H20
  platform. The preflight still passed without OOM/SIGFPE.
- DDP reported that `find_unused_parameters=True` found no unused parameters.
  This is a performance warning only; gradients were finite and checkpoints
  reloaded.

## Final GPU State

After the controller exited, `nvidia-smi` reported GPU0 `28 MiB`, GPU1-GPU7
`1 MiB`, utilization `0%`, and no compute apps.

## Decision

`H20_MINIMAX_BF16_SAFE_READY` is passed for 1-batch MiniMax preflight, including
single-GPU bf16 and DDP8 bf16 one-step train with checkpoint save/reload on
rank0. This does not claim model-quality improvement and does not replace the
official protocol audit.

Next gate: `H20_MINIMAX_PROTOCOL_MATCHES_OFFICIAL` / mismatch / blocked.
