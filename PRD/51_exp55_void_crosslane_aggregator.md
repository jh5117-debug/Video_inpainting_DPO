# Exp55 VOID Cross-Lane Aggregator

Date: 2026-07-01

Branch: `research/exp55-void-crosslane-aggregator-20260701`

Base evidence:

- Exp53B H20: `R1_Q2_T500_S0` and `R2_Q2_T500_S0` both completed checkpoint, strict reload, heldout4 videos, metrics, and visual review.
- Exp54 PAI: R3/R4 SDPO/Linear one-step reports were available from committed branch artifacts; PAI raw evidence was not mounted on H20 and is recorded as `EXP55_PAI_RAW_EVIDENCE_MISSING`.

Decision:

- Status: `EXP55_NO_10STEP_MIXED_ONLY`
- Best H20 candidate: `R1_Q2_T500_S0`
- Best PAI candidate: `R4_Q2_T500_S0`
- Best overall candidate: `R1_Q2_T500_S0`
- One-step PASS candidates: none
- 10-step allowed: no

Scientific position:

VOID remains a VOR-OR inference baseline, a same-model loser-generator candidate, and an adapter-engineering candidate. It is not third-backbone adapter evidence. The current blocker is not runtime/cache/checkpoint health; it is the persistent local-region tradeoff where object/mask improves while overlap, affected, and boundary regions regress and visual review remains mixed.

Next preregistered direction:

Exp56 should run a small one-step local region-safe repair, not a grid and not 10-step. The proposed repair should preserve overlap / affected / boundary more strongly while keeping object-local DPO narrow and loser gradients near zero.
