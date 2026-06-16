# Metric Summary

Status: training_completed_eval_blocked.

Gate2000 completed 2000 training steps, but no DAVIS metric table exists yet.

Reason:

- Upstream VideoPainter eval is not the project fixed raw6 hard-comp protocol.
- Upstream eval does not call the project `inference/metrics.py` backend.
- Upstream eval currently needs additional compatibility work and expects the
  optional `ckpt/flux_inp` path in the DAVIS branch.

Required next step:

Implement an Exp14-only thin DAVIS eval adapter that loads:

```text
VideoPainter baseline branch
VideoPainter + DPO adapter last_weights
```

and then generates raw6 hard-comp outputs for project metric evaluation.
