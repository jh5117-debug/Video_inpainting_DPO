# Exp57 VOID Adaptive Transition-Safe DPO

Date: 2026-07-01

Branch: `research/exp57-void-adaptive-transition-core-20260701`

Base branch: `origin/research/exp55-void-crosslane-aggregator-20260701`

## Goal

Exp57 tests whether VOID can move past the mixed-only one-step regime by protecting transition regions during the one-step adapter update. The experiment is VOID-only and does not change DiffuEraser, VideoPainter, shared trainer code, VOID official source, or `inference/metrics.py`.

## Current Evidence

Exp55 concluded `EXP55_NO_10STEP_MIXED_ONLY`: no Exp53B H20 or Exp54 PAI one-step candidate passed the original gate, so 10-step remained locked.

Exp56-H20 narrowed the blocker. `R5_Q2_T500_S0` removed loser dominance with `loser_contribution_ratio = 0.0`, improved object and outside metrics, but still regressed overlap, affected, and boundary regions. The blocker is now:

`TRANSITION_REGION_DAMAGE_UNDER_OBJECT_LOCAL_UPDATE`

## Exp57 Hypothesis

Fixed loser suppression alone is insufficient. VOID needs an adaptive transition-safe update that:

- suppresses loser gradients when they conflict with winner gradients;
- monitors object, overlap, affected, boundary, and outside loss deltas;
- backtracks or rejects the update when transition regions would worsen;
- preserves outside and transition regions while allowing a narrow object-local preference signal.

## Allowed Work

- Add isolated code in `exp57_void_adaptive_transition_safe/`.
- Use Exp52 cached train4 / heldout4 Q2/T500 tensors.
- Run zero-gap and one-step only.
- Save checkpoints, strict reload them, generate heldout4 videos, compute quadmask-aware metrics, and visually review evidence.

## Forbidden Work

- No VOR-Eval.
- No hard comp.
- No 10-step or longer training.
- No 30/50/100/300/500-step run.
- No VOID official source edits.
- No shared trainer edits.
- No `inference/metrics.py` edits.
- No universal adapter, final SOTA, or third-backbone claim.

## Milestones

| Milestone | Status | Notes |
| --- | --- | --- |
| A | `EXP57_READBACK_DONE` | Reconfirm Exp55/Exp56 mixed-only failure pattern. |
| B | pending | Implement `void_adaptive_transition_safe_dpo_v0`. |
| C | pending | Run zero-gap / adaptive forward sanity. |
| H20-D/E | pending | Run H20 one-step cells only and hand off. |
| PAI-D/E | pending | Run PAI one-step cells only after core branch exists. |

## Scientific Position

VOID remains a VOR-OR inference baseline, same-model loser-generator candidate, and adapter-engineering candidate. It is not third-backbone adapter evidence unless a later one-step gate passes and a later aggregator-approved 10-step gate is genuinely positive on heldout visual and metric evidence.
