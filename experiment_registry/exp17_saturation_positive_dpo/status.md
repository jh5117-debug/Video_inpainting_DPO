# Status

`COMPLETED_NEGATIVE_STAGE1_GATES`

Completed on PAI:

- Exp17a positive Stage1 1000 + DAVIS10 eval.
- Exp17b saturation Stage1 1000 + DAVIS10 eval.
- Exp17c combined Stage1 1000 + DAVIS10 eval.

Decision:

```text
No Exp17 variant beats Exp11 outer b0.75 S2 on DAVIS10.
Do not run Stage1 2000.
Do not run Stage2.
Keep Exp17 as a negative ablation.
```

Best Exp17 variant by metrics: `exp17b_saturation_s1_1000`.
Current overall best remains: `Exp11 boundary outer b0.75 S2`.
