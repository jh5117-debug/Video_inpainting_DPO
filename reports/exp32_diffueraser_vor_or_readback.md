# Exp32 DiffuEraser VOR-OR Readback

Date: 2026-06-27

Status: `EXP32_READBACK_COMPLETED_DATA_GATE_PENDING`

## Git Readback

- branch: `research/exp32-diffueraser-vor-or-2000step-20260627`
- base branch: `origin/research/exp25-vor-or-preference-data`
- HEAD: `bc6dc80206f5e397bda577ba62f9371813e5a657`
- worktree: `/home/hj/H20_Video_inpainting_DPO_exp32_diffueraser_vor_or`
- `git status --short`: clean before this readback
- `git diff --stat`: empty before this readback
- `git diff --check`: passed before this readback

Recent base log:

```text
bc6dc80 Record Exp25 DiffuEraser root-cause matrix
c34db6a Fix Exp25 matrix metric resize
429f7b1 Add Exp25 DiffuEraser matrix review
9edb2b7 Add Exp25 DiffuEraser root-cause matrix runner
81859cc Run Exp25 postpermission asset smoke
63bb306 Record final PAI permission recovery
48be2e9 Record Exp25 PAI postmaintenance blockers
62b77aa Record Exp25 root-cause matrix permission blocker
```

## Source Files Read

- `PRD/47_exp25_vor_or_preference_data.md`
- `experiment_registry/exp25_vor_or_preference_data/status.md`
- `experiment_registry/exp25_vor_or_preference_data/results.tsv`
- `reports/exp25_diffueraser_or_stack_audit.md`
- `reports/exp25_diffueraser_or_stack_audit_v2.md`
- `reports/exp25_diffueraser_primary_stack_decision.md`
- `reports/exp25_diffueraser_or_root_cause_matrix.md`
- `reports/exp25_diffueraser_or_root_cause_matrix_v2.md`
- `/home/hj/H20_Video_inpainting_DPO_exp25_cli4/reports/exp25_gate16_deb_cli4_result.md`
- `/home/hj/H20_Video_inpainting_DPO_exp25_cli4/reports/exp25_gate16_deb_cli4_result.csv`

The Exp25 CLI4 branch was read-only. No files in that worktree were modified.

## VOR Source/Index State

Known from Exp25:

- VOR core archive scope completed: 37/37 files.
- VOR-Eval was fully extracted and is final/held-out only.
- VOR-Train and VOR-Train-MASK archive continuity and semantic pairing passed
  earlier audits.
- Gate32 canonical raw generation completed but quality yield was poor.

Unknown for Exp32:

- no verified train32 + heldout16 scene-disjoint VOR-OR preference manifest is
  present in this branch.
- no verified DE-B Gate64 train32 + heldout16 split is present in this branch.
- no verified controlled-corruption Gate64 split is present in this branch.

## DiffuEraser Stack State

The latest useful stack evidence comes from the CLI4 Gate16 run:

```text
DE-B_sft_raw6_d8_propainter
pcm_mode = none
prior_mode = propainter
no_pcm_steps = 6
guidance = 0.0
mask_dilation_iter = 8
hard_comp = false
condition = V_obj
winner = V_bg
mask = object foreground mask
loser = DiffuEraser raw output
```

Gate16 result:

| bucket | count |
| --- | ---: |
| technical valid | 16 |
| medium-hard eligible | 7 |
| hard but plausible | 7 |
| too close | 0 |
| trivial bad | 2 |
| technical invalid | 0 |

Decision from CLI4: `EXP25_DIFFUSERASER_GATE16_PASSED`.

## Data Gate Decision

Exp32 cannot start Stage1/Stage2 2000-step training yet.

Reason:

- Gate16 has only 16 generated rows.
- The required training gate needs at least train32 + heldout16 scene-disjoint
  rows and a Gate64-level yield check.
- VOR-Eval is held-out only and cannot be used for training, loser mining,
  threshold tuning, or checkpoint choice.

Allowed next action:

- build or read a scene-disjoint source pool for a DE-B Gate64-style data gate;
- generate candidates only into Exp32 output roots;
- review all candidate videos and write
  `reports/exp32_vor_or_data_gate.md` before any training.

Forbidden next action:

- no Stage1/Stage2 training before data gate;
- no MiniMax adapter run;
- no EffectErase output as primary DPO loser;
- no hard comp as primary loser.

## Right-Side Protection

Read-only PAI checks:

- `2026-06-27T12:56:38+08:00`: GPUs 0-7 idle, no compute apps.
- `2026-06-27T12:58:03+08:00`: GPUs 0-7 idle, no compute apps.
- `2026-06-27T13:04:42+08:00`: no active Exp30/MiniMax process and no compute
  apps.

Protected state:

- Exp30 worktree exists locally at
  `/home/hj/H20_Video_inpainting_DPO_exp30_vor_or_minimax`.
- Exp30/MiniMax outputs exist on PAI.
- stale MiniMax locks reserve GPU0 and GPU5.

Exp32 planned GPU for data/smoke work is GPU2. GPU0 and GPU5 remain avoided.

No signal was sent and no right-side file was modified.

## Decision

Readback passes. Exp32 is data-gate pending and training-blocked.

