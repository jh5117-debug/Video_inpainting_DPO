# Exp34 Objective Ablation Prep Readback

Date: 2026-06-27

Status: `EXP34_READBACK_COMPLETED_OBJECTIVE_PREP_PENDING`

## Git Readback

- branch: `research/exp34-objective-ablation-prep-postminimax-20260627`
- base branch: `origin/research/exp27-paper-grounded-preference-study`
- HEAD: `17b99a421fbf1bb79a446713a1f30ef6c8ecc769`
- worktree: `/home/hj/H20_Video_inpainting_DPO_exp34_objective_prep`
- `git status --short`: clean before this readback
- `git diff --stat`: empty before this readback
- `git diff --check`: passed before this readback

Recent base log:

```text
17b99a4 Run Exp27 Linear-DPO true-model micro gate
265bfe0 Record Exp27 true-model SDPO gate
e708715 Add Exp27 true-model tiny-step gate
4aed0d8 Add Exp27 true model objective parity gate
005e603 Record final PAI permission recovery
2b1a491 Record Exp27 PAI postmaintenance blockers
f73e542 Scan Exp27 SDPO real residual distribution
3d95c9e Record Exp27 PAI persistence completion
```

## Source Files Read

- `PRD/49_exp27_paper_grounded_preference_study.md`
- `experiment_registry/exp27_paper_grounded_preference_study/status.md`
- `experiment_registry/exp27_paper_grounded_preference_study/results.tsv`
- `reports/exp27_true_model_objective_parity_readback.md`
- `reports/exp27_sdpo_true_model_forward_parity.md`
- `reports/exp27_linear_true_model_10step_readback.md`
- `reports/exp27_linear_true_model_10step.md`
- `reports/exp27_localdpo_official_path_fix.md`
- `reports/exp27_nontrivial_parity_and_localdpo_smoke_20260624.md`
- `exp27_paper_grounded_preference_study/code/localdpo_compat.py`
- `exp27_paper_grounded_preference_study/code/localdpo_full_adapter.py`
- `exp27_paper_grounded_preference_study/code/official_parity.py`
- `exp27_paper_grounded_preference_study/scripts/run_exp27_true_model_objective_parity.py`
- `exp27_paper_grounded_preference_study/scripts/run_exp27_linear_true_model_10step.py`

## Exp27 Completed State

SDPO true-model:

- status: `TRUE_MODEL_PARITY`
- records: `256`
- lambda max abs diff: `0.0`
- loss max abs diff: `0.0`
- output-gradient cosine min: `0.9999998807907104`
- S1 lambda<1 ratio: `0.25`
- tiny-step status: `passed`

Linear-DPO true-model:

- status: `LINEAR_TRUE_MODEL_1_10_STEP_PASSED`
- variants: `linear_frozen`, `linear_ema`
- steps: `10`
- Linear-Frozen max grad norm: `0.48048678696353975`
- Linear-Frozen step10 policy delta norm: `0.0012969709135074255`
- Linear-Frozen reference delta norm: `0.0`
- Linear-EMA max grad norm: `0.49775458360972635`
- Linear-EMA step10 policy delta norm: `0.0013002275364513564`
- Linear-EMA reference delta norm: `1.819002953296671e-08`

LocalDPO:

- official mask digest path fix passed;
- six-video corruption pair and original loss 1/10-step smoke passed;
- faithful 24F adaptation remains pending.

Not started:

- O0-O5 controlled objective study;
- RC-FPO;
- any 50-step objective run.

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

Exp34 is CPU-only at readback. No signal was sent and no right-side file was
modified.

## Decision

Readback passes. Exp34 may prepare CPU-only objective config scaffolding and
guardrail tests. It must not launch O0-O5, LocalDPO 24F, RC-FPO, MiniMax, or any
GPU objective study until the right-side MiniMax work is explicitly completed
or released.

