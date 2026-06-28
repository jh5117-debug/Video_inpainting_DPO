# Exp39 H20 MiniMax Mirror Readback

Date: 2026-06-28

Status: `EXP39_H20_MIRROR_READBACK_COMPLETED_H20_GIT_SYNC_BLOCKED`

## Scope

This readback starts the H20 mirror and bf16/SIGFPE debug track for MiniMax.
It does not launch training and does not modify PAI. PAI was used only for
read-only GPU/process inspection.

## Git And H20 Access

- H20 host: `ubuntu@27.190.15.128`.
- H20 hostname observed: `instance-afs92r3e`.
- H20 GPUs observed at connection time: 8 x NVIDIA H20, idle.
- H20 old repo: `/home/nvme01/H20_Video_inpainting_DPO`.
- Intended H20 worktree:
  `/home/nvme01/H20_Video_inpainting_DPO_exp39_minimax_h20`.
- H20 old repo branch at audit time: `main`.
- H20 old repo HEAD at audit time:
  `d38ec2ded8135027d68ed379044c6986eb9937ed`.
- Dirty audit was preserved in the H20 old repo under:
  `reports/exp39_h20_initial_dirty_audit.md`,
  `reports/exp39_h20_initial_dirty_audit.patch`, and
  `reports/exp39_h20_initial_dirty_status.txt`.

Because the H20 old repo was dirty/old and its `external/VBench` submodule made
plain status checks fragile, the old worktree was not checked out, reset, or
cleaned.

## Base Readback

- Exp39 branch: `research/exp39-h20-minimax-mirror-bf16-20260628`.
- Base branch:
  `origin/research/exp38-minimax-full-adapter-breakthrough-20260628`.
- Current base HEAD:
  `06b17c0a4be2cb82d1ffbdf7b6c93406f37a3ff8`
  (`Run Exp38 MiniMax SFT-DPO rescue gate`).
- Local readback worktree:
  `/home/hj/H20_Video_inpainting_DPO_exp39_minimax_h20_local`.

H20 GitHub object transfer was unreliable during this milestone:

- `git fetch --all --prune` was stopped after stalling.
- A single-branch full/shallow H20 checkout was too slow.
- A broad sparse checkout also timed out while fetching blobs.
- The incomplete H20 clone attempts were stopped and removed after verifying
  they were created by this Exp39 session.

This blocks declaring the H20 worktree ready. It does not affect the source
readback, because the same current Exp38 remote HEAD was read locally from
GitHub and used for this Exp39 branch.

## PAI Protection Readback

PAI read-only audit observed compute processes on GPU2/GPU3/GPU4 and the legacy
MiniMax wrapper command. No PAI signal was sent, no PAI GPU was used, and no PAI
file was modified.

The next PAI interaction is limited to read-only asset inventory and checksum
generation.

## MiniMax Current Scientific State

Exp30:

- Status: `MINIMAX_ADAPTER_RECIPE_NOT_READY`.
- Gate64 V3 pool ready, train32/heldout16 locked.
- Frozen/EMA 10-step recipes passed plumbing checks but had visual better
  `0/32` and slightly negative local/outside metrics.

Exp35:

- Status: `MINIMAX_RESCUE_RECIPE_NOT_READY`.
- Hard-noise R1/R2/R3 10-step recipes produced movement but no heldout quality
  improvement.
- Visual better `0/48`; 30-step remains locked.

Exp36:

- Final status: `MINIMAX_PLUMBING_POSITIVE_RECIPE_NOT_READY`.
- Checkpoint loading and inference sensitivity passed.
- Winner-SFT can lower train loss and move outputs, but heldout visual better
  remained `0/24`.

Exp37:

- Status: `MINIMAX_LOCALDPO_BADNOISE_PARETO_MIXED`.
- LocalDPO-style data and outside-sane bad-noise states were built.
- R1/R2/R3 each had only `1/16` visually better heldout rows.
- Paper role remains `TWO_BACKBONE_PLUS_MINIMAX_PLUMBING_ONLY`.

Exp38:

- Latest status: `MINIMAX_SFT_DPO_RESCUE_10STEP_NEGATIVE`.
- LocalDPO v2 filtered pool and bad-noise v2 states were built.
- SFT-DPO 10-step rescue remained negative.
- 30-step remains locked.

## Required Answers

1. H20 current repo is not latest. It is an old dirty `main` checkout and must
   be preserved.
2. H20 must use an isolated Exp39 worktree/clone. That worktree is not ready
   yet because GitHub object transfer from H20 timed out.
3. H20 dirty files were preserved; no reset, clean, or checkout was performed
   in the old repo.
4. Current MiniMax scientific status is plumbing-positive but quality-negative;
   Exp38 latest rescue is negative.
5. Required migration assets are the locked Exp30 Gate64 V3 manifests, Exp37
   LocalDPO train/heldout and bad-noise state manifests, Exp38 LocalDPO v2
   filtered manifests, bad-noise v2 states, and any selected checkpoints needed
   for H20 smoke/replay.
6. Do not migrate full VOR archives, full logs, old failed visual caches,
   EffectErase VOR full data, or unrelated VideoPainter outputs.
7. H20 weights and conda environments are not yet audited in this milestone.
8. Do not start H20 MiniMax training, PAI MiniMax training, 30-step MiniMax,
   RC-FPO, EffectErase adapter, or any universal-adapter experiment.
9. BF16/SIGFPE debug plan: audit H20 CUDA/Torch/env first, reproduce the
   smallest MiniMax smoke, prefer dtype/runtime/config fixes, and isolate any
   required code patch to Exp39 only.
10. This milestone updates Exp39 PRD, registry, and readback report. Asset
    inventory and H20 env/weight audit are the next milestones.

## Decision

```text
EXP39_H20_MIRROR_READBACK_COMPLETED_H20_GIT_SYNC_BLOCKED
```

The H20 mirror lane is valid to continue with small-file asset inventory and
environment audit, but H20 is not ready for MiniMax training or bf16 claims.
