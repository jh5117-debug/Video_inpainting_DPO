# Exp45 Targeted Pair Scale-Up Mining

Status: `MINIMAX_TARGETED_SCALEUP_BLOCKED_SOURCE_ROOT_UNAVAILABLE`

## Scope

Milestone C was scheduled to continue Exp44 targeted same-source mining on PAI
only. The intended goal was to expand the partial `24/8/8` Stage2 handoff to
at least `32/16/16`, preferably `64/24/24`, using official MiniMax raw
inference and strict no-training boundaries.

## Pre-Mining Readback

Read before deciding whether mining was safe:

- `PRD/59_exp45_pai_minimax_pair_scaleup.md`
- `experiment_registry/exp45_pai_minimax_pair_scaleup/status.md`
- `reports/exp45_h20_handoff_package.md`
- `reports/exp44_source_group_plan.json`
- `reports/exp44_targeted_visual_relabel_summary.json`
- `reports/exp44_same_source_pair_summary.json`

Exp44 source state:

- current same-source pairs: `40`
- current split: `24/8/8`
- formal minimum target: `32/16/16`
- preferred target: `64/24/24`
- pair groups: `10`
- search groups: `2`
- shadow groups: `2`
- bad-noise v4 usable H-states: `26`

## Blocker

The current session cannot access the required PAI/NAS roots:

- `/mnt/nas`: missing
- `/mnt/workspace`: missing
- requested Exp44 source root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining/targeted_mining_20260629_161742`
- requested output root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp45_pai_minimax_pair_scaleup`

Because the source videos, masks, GT frames, raw MiniMax outputs, and target
output roots are unavailable, running official MiniMax inference would not be
reproducible or safe. No GPU task was launched.

## What Was Not Done

- H20 touched: `false`
- H20 GPU used: `false`
- H20 output written: `false`
- MiniMax inference launched: `false`
- SFT/DPO training launched: `false`
- optimizer step: `false`
- VOR-Eval used: `false`
- hard comp used: `false`
- empty candidate manifest fabricated: `false`

## Required Next Action

Resume Milestone C only from a true PAI/NAS-mounted session where:

- `/mnt/nas` exists;
- `/mnt/workspace` exists;
- Exp44 targeted mining source root is readable;
- Exp45 output root can be written;
- GPU0/GPU1 are available or explicitly assigned.

Then run targeted mining on the preregistered A/B/C/D source groups, with no
training and no optimizer step.
