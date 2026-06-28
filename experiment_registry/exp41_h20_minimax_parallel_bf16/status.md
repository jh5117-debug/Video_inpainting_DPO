# Exp41 Status

Current status: `H20_MINIMAX_BF16_SAFE_READY`

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

## 2026-06-29 Data / Weight Audit

Current status: `H20_MINIMAX_DATA_READY`

- H20 mirror active-path validation passed for Exp30/37/38 and Exp41/Exp40
  LocalDPO manifests: `2242` refs checked, `0` missing.
- Exp40 H20-safe manifests are present for LocalDPO v3 train/search/shadow:
  `64/24/24` rows.
- Missing Exp40 evidence files were filled from PAI read-only: `224` files,
  `333063059` bytes.
- Missing legacy evidence files were filled from PAI read-only: `232` files,
  `237644031` bytes.
- H20 decode audit passed for `112` Exp40 raw outputs and de-duplicated
  source/winner/mask mp4s.
- MiniMax scheduler/transformer/VAE weights resolve under the H20 `current`
  symlink.
- This is not a quality-positive MiniMax result; BF16/SIGFPE and official
  protocol gates remain pending.

Reports:

- `reports/exp41_h20_minimax_data_audit.md`
- `reports/exp41_h20_minimax_data_audit.csv`
- `reports/exp41_h20_minimax_manifest_validation.csv`
- `reports/exp41_h20_minimax_missing_assets.csv`
- `reports/exp41_h20_minimax_decode_audit.csv`

## 2026-06-29 BF16 / SIGFPE Preflight

Current status: `H20_MINIMAX_BF16_SAFE_READY`

- P0 torch bf16 matmul/backward: PASS.
- P1 VAE fp32 encode/decode: PASS.
- P2 DiT bf16 forward no grad: PASS.
- P3 DiT bf16 forward/backward with fp32 loss: PASS.
- P4 MiniMax fp32 one-batch train + checkpoint reload: PASS.
- P5 MiniMax bf16-safe single-GPU one-batch train + checkpoint reload: PASS.
- P6 MiniMax bf16-safe DDP2 one-batch train + rank0 checkpoint reload: PASS.
- P7 MiniMax bf16-safe DDP8 one-batch train + rank0 checkpoint reload: PASS.
- No SIGFPE, OOM, CUDA error, NaN/Inf, or Xid was observed.
- Final H20 GPU0-GPU7 compute apps: none.

Reports:

- `reports/exp41_h20_bf16_preflight.md`
- `reports/exp41_h20_bf16_preflight.csv`
- `reports/exp41_h20_bf16_preflight_summary.json`
- `reports/exp41_h20_bf16_preflight_rank_details.csv`
