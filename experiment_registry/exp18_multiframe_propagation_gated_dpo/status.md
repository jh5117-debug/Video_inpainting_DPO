# Status

```text
PAI_GATE_COMPLETED_NEGATIVE_ABLATION
```

Completed on PAI:

- limit=100 multi-frame propagation cache
- Exp18a Stage1-500
- Exp18b Stage1-500
- Exp18c oracle Stage1-500 diagnostic
- DAVIS10 metric sanity
- visual judgement
- dpo_diag summaries

Decision:

```text
No Exp18 variant beats Exp11 outer b0.75 S2.
Do not run Stage1 1000, full cache, Stage1 2000, or Stage2 under the current formulation.
```
