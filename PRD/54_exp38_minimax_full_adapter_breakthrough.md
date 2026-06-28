# PRD 54: Exp38 MiniMax Full Adapter Breakthrough

Date: 2026-06-28

Status: `EXP38_READBACK_COMPLETED`

Exp38 is an isolated MiniMax rescue track based on
`origin/research/exp37-minimax-localdpo-badnoise-rescue-20260627`.

## Scope

Exp38 investigates why MiniMax remains plumbing-positive but not
quality-positive after Exp30, Exp35, Exp36, and Exp37.

Allowed:

- root-cause readback and failure taxonomy;
- official MiniMax protocol audit;
- train-overfit diagnosis;
- bounded 10-step recipe exploration after preregistration;
- paper evidence packaging for DiffuEraser and VideoPainter.

Forbidden:

- blind 30-step;
- 2000-step MiniMax training;
- RC-FPO;
- modifying `inference/metrics.py`;
- modifying shared trainer;
- overwriting Exp30/35/36/37 results;
- universal-adapter, all-models-supported, final-SOTA, or top-conference
  novelty claims.

## Readback Summary

Branch:

```text
research/exp38-minimax-full-adapter-breakthrough-20260628
```

Start HEAD:

```text
558c2f263469f4ee6ee46e2a1b26a8082515dded
```

Base:

```text
origin/research/exp37-minimax-localdpo-badnoise-rescue-20260627
```

Exp30/35/36/37 conclusions read:

- Exp30 Gate64 MiniMax frozen/EMA 10-step: `0/32` visual better; near-tie or
  slightly negative heldout metrics.
- Exp35 hard-noise rescue: `0/48` visual better; utility scale/update signal
  was too weak or harmful.
- Exp36 ruled out checkpoint/load and ignored-weight failure. Identity replay
  was deterministic and a 1.01x transformer perturbation changed outputs.
- Exp36 winner-SFT reduced train loss and moved outputs but had `0/24`
  heldout visual better rows.
- Exp37 LocalDPO-style corruption and bad-noise states were built cleanly, but
  the preregistered 10-step LocalDPO-badnoise recipes reached only `1/16`
  visible heldout improvement per recipe.

Current MiniMax status:

```text
MINIMAX_PLUMBING_POSITIVE_RECIPE_NOT_READY
```

Paper language:

```text
TWO_BACKBONE_PLUS_MINIMAX_PLUMBING_ONLY
```

## PAI GPU Readback

PAI host:

```text
dsw-753014-85f54df947-bkp7h
```

Two checks showed:

- GPU0: `0 MiB`, `0%`, no compute PID.
- GPU1: `0 MiB`, `0%`, no compute PID.
- GPU2-GPU7: occupied by `/usr/local/bin/python3.10` jobs at about
  `136 GiB` each and `100%` utilization.

Legacy `runtime/cli4` lock files still include GPU1/GPU2/GPU3/GPU4 entries.
No lock, process, or heartbeat was modified. No signal was sent.

PAI permissions:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch` is writable by
  `hj`, and the Exp38 log directory was created.
- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo` is not writable by
  `hj` during readback. Exp38 training/output milestones must either receive a
  minimal root permission fix for the Exp38 experiment directory or write
  non-checkpoint runtime assets under the writable log root until fixed.

## Next Milestones

1. Exp38 failure taxonomy and decision tree.
2. MiniMax train-overfit diagnosis.
3. Official MiniMax reproduction/protocol audit.
4. LocalDPO-style pool v2 and bad-noise v2 only if justified.
5. Preregistered bounded rescue recipes.
6. 10-step quality-positive gate.
7. Conditional 30-step only after a strict 10-step pass.
8. DiffuEraser + VideoPainter evidence pack, keeping protocols separate.

No GPU task or training was launched by this readback milestone.

