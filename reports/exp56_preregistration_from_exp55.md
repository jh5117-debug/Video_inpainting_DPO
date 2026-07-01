# Exp56 Preregistration From Exp55

Do not execute this plan in Exp55.

Experiment: `EXP56 VOID LOCAL REGION-SAFE OBJECTIVE REPAIR`

Goal: fix the observed pattern where object/mask improves but overlap, affected, and boundary regress.

Allowed scope:

- Q2/T500 only.
- train4/heldout4 first.
- one-step only.
- no VOR-Eval.
- no hard comp.
- no 10-step unless a later one-step PASS is achieved.

R5:

- object-only DPO.
- no affected DPO push.
- affected / overlap / boundary as preservation penalties.
- loser_grad_scale = 0.0.
- winner_anchor >= 0.10.
- outside preservation >= 0.10.
- boundary preservation >= 0.15.
- affected preservation >= 0.10.
- overlap preservation >= 0.15.
- local-only margin.
- half-step or reduced LR.

R6:

- object + affected DPO.
- affected gradient clipped.
- boundary and overlap preservation stronger than Exp53B.
- loser_grad_scale <= 0.02.
- loser_gap_clip_tau tighter than R2 if used.

Suggested execution after approval:

- H20: R5_Q2_T500_S0 and R5_HALF_Q2_T500_S0.
- PAI: R6_Q2_T500_S0 and best Exp54-safe variant.

No universal adapter, no final SOTA, no third-backbone claim.
