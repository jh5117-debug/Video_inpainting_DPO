# Exp41 Status

Current status: `EXP41_H20_MINIMAX_PARALLEL_READBACK_COMPLETED`

## 2026-06-29 Readback / GPU Release

- Branch: `research/exp41-h20-minimax-parallel-bf16-20260629`.
- Start HEAD: `ecd82ef8bfefd1efba063d2a240631c1b7230b1d`.
- Base: `origin/research/exp40-minimax-psnr-safe-rescue-20260628`.
- H20 worktree:
  `/home/nvme01/H20_Video_inpainting_DPO_exp41_minimax_h20_parallel`.
- HAL/local worktree:
  `/home/hj/H20_Video_inpainting_DPO_exp41_minimax_h20_parallel`.
- Exp39 mirror/env reports were read from the Exp39 branch/worktree because
  Exp41 is based on Exp40, which does not contain the Exp39 mirror commit.
- H20 GPU4 non-system compute PGID `3365988` was terminated with TERM after
  audit; no KILL was required.
- Final H20 GPU0-GPU7 compute apps: none.
- PAI was used read-only only; no PAI GPU, signal, output mutation, or runtime
  mutation occurred.

Next milestones:

1. `H20_MINIMAX_DATA_READY` audit.
2. BF16/SIGFPE safe runtime preflight.
3. Official MiniMax protocol audit.
4. Gated SFT/DPO lanes only if prerequisites pass.

Reports:

- `reports/exp41_h20_minimax_parallel_readback.md`
- `reports/exp41_h20_gpu_release_audit.md`
