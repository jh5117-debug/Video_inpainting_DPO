# Exp41 H20 MiniMax Parallel BF16

Status: `H20_MINIMAX_BF16_SAFE_READY`

Exp41 is an H20-only parallel MiniMax adapter track. PAI remains read-only and
continues to own any active Exp40/PAI-side work. This branch does not modify
MiniMax official source files, `inference/metrics.py`, shared trainers, or
Exp1-Exp40 history.

## Scope

- Use H20 as a second server for MiniMax adapter debugging and gated training.
- Prefer existing code, flags, environment variables, launch wrappers, and run
  configs over source changes.
- If a BF16/SIGFPE fix requires source edits, stop and write a patch proposal
  instead of editing source.
- Do not write universal-adapter, final-SOTA, all-models-supported, or
  top-conference-novelty claims.

## Source Of Truth

- Branch: `research/exp41-h20-minimax-parallel-bf16-20260629`.
- Base: `origin/research/exp40-minimax-psnr-safe-rescue-20260628`.
- Start HEAD: `ecd82ef8bfefd1efba063d2a240631c1b7230b1d`.
- H20 worktree:
  `/home/nvme01/H20_Video_inpainting_DPO_exp41_minimax_h20_parallel`.
- HAL/local worktree:
  `/home/hj/H20_Video_inpainting_DPO_exp41_minimax_h20_parallel`.

## Current MiniMax State

- Exp30/35/36/37/38 show MiniMax is trainable and inference uses trained
  weights, but no heldout quality-positive adapter result exists.
- Exp38 R1 had a small raw PSNR signal but failed boundary/outside and visual
  safety.
- Exp40 built a VOR-Train LocalDPO v3 minimum pool:
  `train64/search24/shadow24`, zero split scene overlap, all selected rows
  `MEDIUM_HARD_ELIGIBLE`.
- Exp40 Step0 baseline is established on the LocalDPO v3 minimum pool.
- Exp39 H20 mirror/env smoke passed: required H20 MiniMax training/smoke paths
  are present, weights resolve, and H20 `wan` env imports MiniMax.

## H20 GPU Release

Initial H20 GPU audit found GPU4 occupied by a non-system compute task:

```text
PID 3365990, PGID 3365988
/home/nvme03/workspace/lingbot-world/.conda_envs/lingbot-world-v2/bin/python
-m cam_physgeo.dpo.failure_diagnostics sigma-sensitivity ...
```

The task was unrelated to Video_inpainting_DPO and was cleared with
`TERM -- -3365988`; no `KILL` was required. Final H20 GPU state has no compute
apps on GPU0-GPU7. `nvitop` still holds `/dev/nvidia*` handles but is not a
compute process.

Report:

- `reports/exp41_h20_gpu_release_audit.md`
- `reports/exp41_h20_gpu_release_audit.csv`


## H20 Data / Weight Audit

Status: `H20_MINIMAX_BF16_SAFE_READY`

H20 mirror validation passed after filling missing evidence assets from PAI
read-only rsync. The final audit checked `2242` active refs across Exp30, Exp37,
Exp38, and Exp41/Exp40 manifests with `0` missing. Exp40 H20-safe LocalDPO v3
manifests are available for `64` train, `24` search, and `24` shadow rows.

Additional checks:

- Exp40 raw output decode: `112/112` readable.
- Exp40 direct VOR condition/winner/mask decode: `22/22` each readable.
- First-frame mask non-empty check: `0` empty masks.
- MiniMax scheduler/transformer/VAE files resolve via H20 `current` symlink.

This gate does not authorize training by itself. BF16/SIGFPE runtime preflight
and official protocol audit remain pending.

Reports:

- `reports/exp41_h20_minimax_data_audit.md`
- `reports/exp41_h20_minimax_manifest_validation.csv`
- `reports/exp41_h20_minimax_missing_assets.csv`
- `reports/exp41_h20_minimax_decode_audit.csv`


## BF16 / SIGFPE Runtime Preflight

Status: `H20_MINIMAX_BF16_SAFE_READY`

Exp41 ran P0-P7 on H20 using an Exp41-only helper and launcher. The helper kept
VAE encode/decode in fp32, ran MiniMax DiT bf16 where required, reduced losses
in fp32, disabled flash/memory-efficient SDPA where PyTorch exposes backend
toggles, and used xFormers/flash-attn disable env flags.

All cases passed: P0 torch bf16 backward, P1 VAE fp32 encode/decode, P2/P3 DiT
bf16 forward/backward, P4 fp32 one-batch train, P5 bf16-safe single-GPU train,
P6 bf16-safe DDP2 train, and P7 bf16-safe DDP8 train. Rank0 checkpoint
save/reload passed for P4-P7. No SIGFPE, OOM, CUDA error, NaN/Inf, or Xid was
observed. Final H20 GPU compute apps: none.

This is a runtime gate only. It does not claim MiniMax quality improvement and
does not replace the official MiniMax protocol audit.

Reports:

- `reports/exp41_h20_bf16_preflight.md`
- `reports/exp41_h20_bf16_preflight.csv`
- `reports/exp41_h20_bf16_preflight_summary.json`
- `reports/exp41_h20_bf16_preflight_rank_details.csv`

## Readback Decision

H20 can proceed to data/weight completeness audit and BF16/SIGFPE preflight
before any MiniMax training.

Next required statuses:

```text
H20_MINIMAX_DATA_READY or H20_MINIMAX_DATA_PARTIAL/BLOCKED
H20_MINIMAX_BF16_SAFE_READY or H20_MINIMAX_FP32_ONLY_READY/BLOCKED
H20_MINIMAX_PROTOCOL_MATCHES_OFFICIAL or MISMATCH/BLOCKED
```

No SFT/DPO/500-step lane is authorized until those gates complete.
