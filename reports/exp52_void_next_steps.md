# Exp52 VOID Next Steps

Recommended next minimal experiment:

1. Generate full Step0/Step1 heldout evidence for `R1_Q2_T500_S0`.
2. Run one targeted nearby one-step ablation on Q1/Q2 at T300 or T500.
3. Require affected/overlap safety before any 10-step.
4. Keep `proj_out` first; do not escalate to LoRA unless S0 is stable but too weak.
5. No VOR-Eval, no hard comp, no 30/50/100-step training.

Stop condition: if Q2/Q1 one-step also damages affected/overlap, keep VOID as baseline/loser generator and resume third-model search elsewhere.
