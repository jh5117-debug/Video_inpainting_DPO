# Exp41 H20 MiniMax Parallel BF16

Status: `EXP41_H20_MINIMAX_PARALLEL_READBACK_COMPLETED`

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
