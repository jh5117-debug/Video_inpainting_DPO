# Status

Current status:

```text
PAI_GATE_COMPLETED_NEGATIVE_ABLATION
```

Completed:

- Exp18 folder and registry created.
- Multi-frame propagation cache code added.
- Exp18 dataset/loss/diagnostics/training entry added.
- PAI guarded launch scripts added.
- limit=100 propagation cache on PAI.
- propagation quality report from real data.
- Exp18a/b/c Stage1-500 gates.
- DAVIS10 visual/metric judgement.
- true DAVIS10 hybrid eval script added for SFT / Exp11 / Exp18 comparisons.

Decision:

```text
No Exp18 variant beats Exp11 outer b0.75 S2.
Do not expand to Stage1 1000, full cache, Stage1 2000, or Stage2 under the current formulation.
```
