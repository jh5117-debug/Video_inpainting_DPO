# Status

```text
DAVIS10_EVAL_COMPLETED_NEGATIVE_GATE
```

Exp19 is isolated from old experiments. The unsafe shared `additional_residuals`
interfaces were replaced by an Exp19-only hook wrapper. Exp19b trained for 500
adapter-only Stage2 steps and the legal inference wrapper now loads the external
flow adapter and passes real DAVIS completed-flow context through DiffuEraser
windows.

DAVIS10 completed, but the positive gate failed: Exp19b is essentially tied
with Exp11 and does not meet the required temporal improvement threshold. Do not
expand to 1000 steps, DAVIS50, full cache, or full training.
