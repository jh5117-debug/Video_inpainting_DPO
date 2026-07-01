# Exp51 VOID Next Steps

Recommended next minimal action:

`NEXT_VOID_R1_ROW0_SMOKE_THEN_RECIPE_BY_RECIPE_TRAIN4`

1. Run R1-only row0 forward to force a checkpoint within a short bounded runtime.
2. If checkpoint appears, run R1 train4 one-step and heldout4 video evidence.
3. Only if R1 one-step passes, run R1 10-step.
4. Then repeat R2/R3/R4 one at a time.
5. Prefer Q1 object-only and Q2 strict affected before Q3 broad affected.

Do not run long VOID training and do not promote VOID as third evidence.
